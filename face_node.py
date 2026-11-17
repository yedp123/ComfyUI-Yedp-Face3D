import base64
import io as py_io
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from server import PromptServer

from .server_routes import cache_driving_batch


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


def build_phase1_placeholder_passes(
    front_image: torch.Tensor,
    driving_video: torch.Tensor | None,
    depth_weight: float,
    render_width: int,
    render_height: int,
    turntable_frames: int,
    mask_feather: int,
    anatomical_guides: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate zero-VRAM CPU placeholder passes matching final Phase 4 shapes and dtypes."""
    batch_size = int(driving_video.shape[0]) if driving_video is not None else int(turntable_frames)
    batch_size = max(1, batch_size)

    ys = torch.linspace(-1.0, 1.0, render_height, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, render_width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Elliptical head silhouette + optional minimal neck stub for Phase 1 pipeline wiring tests.
    head_r2 = (grid_x / 0.38) ** 2 + ((grid_y + 0.06) / 0.52) ** 2
    inside_head = head_r2 <= 1.0

    if anatomical_guides:
        inside_neck = (grid_x.abs() <= 0.16) & (grid_y >= 0.36) & (grid_y <= 0.85)
        silhouette = (inside_head | inside_neck).to(torch.float32)
    else:
        silhouette = inside_head.to(torch.float32)

    z_surface = torch.sqrt((1.0 - head_r2.clamp(max=1.0)).clamp(min=0.0))
    depth_scale = max(0.0, min(depth_weight, 2.5))
    depth_single = (z_surface * (0.55 + 0.35 * min(depth_scale, 1.5)) + 0.1 * inside_head.to(torch.float32)).clamp(0.0, 1.0)
    if anatomical_guides:
        neck_only = (~inside_head) & (silhouette > 0.5)
        depth_single = torch.where(neck_only, torch.full_like(depth_single, 0.25), depth_single)

    nx = (grid_x / 0.38) * silhouette
    ny = (-(grid_y + 0.06) / 0.52) * silhouette
    nz = z_surface.clamp(min=0.1) * silhouette + (1.0 - silhouette)
    norm_vec = torch.stack((nx, ny, nz), dim=-1)
    norm_vec = F.normalize(norm_vec, p=2, dim=-1)
    normal_single = torch.where(
        silhouette.unsqueeze(-1) > 0.5,
        (norm_vec * 0.5) + 0.5,
        torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32),
    )

    # Sample average front skin tone for the Phase 1 shaded placeholder without GPU work.
    avg_color = front_image[0].detach().to("cpu", dtype=torch.float32).mean(dim=(0, 1)).clamp(0.15, 0.95)
    key_light = F.normalize(torch.tensor([0.45, 0.55, 0.70], dtype=torch.float32), dim=0)
    lambert = (norm_vec * key_light).sum(dim=-1).clamp(min=0.0)
    shade = (0.28 + 0.72 * lambert).unsqueeze(-1)
    bg_color = torch.tensor([0.08, 0.09, 0.11], dtype=torch.float32)
    lit_single = torch.where(
        silhouette.unsqueeze(-1) > 0.5,
        (avg_color.view(1, 1, 3) * shade).clamp(0.0, 1.0),
        bg_color.view(1, 1, 3),
    )

    depth_single_rgb = depth_single.unsqueeze(-1).repeat(1, 1, 3)

    lit_batch = lit_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    depth_batch = depth_single_rgb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    normal_batch = normal_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    raw_mask_batch = silhouette.unsqueeze(0).repeat(batch_size, 1, 1)
    feathered_mask = feather_mask(raw_mask_batch, mask_feather)

    if batch_size > 1 and driving_video is None:
        # Subtle brightness pulse across turntable placeholder frames to confirm batch order.
        angles = torch.linspace(-math.pi / 2, math.pi / 2, batch_size, dtype=torch.float32)
        mod = (0.88 + 0.12 * torch.cos(angles)).view(batch_size, 1, 1, 1)
        lit_batch = (lit_batch * mod).clamp(0.0, 1.0)

    return lit_batch, depth_batch, normal_batch, feathered_mask


class YedpFace3D(io.ComfyNode):
    """ComfyUI V3 node for multi-view 3D face reconstruction, mocap, relighting, and multi-pass export."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YedpFace3D",
            display_name="Yedp Face3D",
            category="Yedp/Face3D",
            description=(
                "Reconstructs a canonical 478-vertex 3D face on CPU, animates via ARKit/video/webcam, "
                "relights interactively in Three.js, and exports 4 synchronized ControlNet passes."
            ),
            search_aliases=["face3d", "3d face", "mediapipe", "relight", "face mocap"],
            inputs=[
                io.Image.Input(
                    "front_image",
                    tooltip="Primary front-facing reference portrait.",
                ),
                io.Image.Input(
                    "three_quarter_image",
                    optional=True,
                    tooltip="Optional 3/4 view photo for lateral cheek depth and texture blending.",
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
                    tooltip="Output render width for all 4 exported passes.",
                ),
                io.Int.Input(
                    "render_height",
                    default=512,
                    min=128,
                    max=2048,
                    step=8,
                    tooltip="Output render height for all 4 exported passes.",
                ),
                io.Int.Input(
                    "turntable_frames",
                    default=24,
                    min=1,
                    max=240,
                    step=1,
                    tooltip="Frame count for the -90 to +90 deg turntable sweep when no driving video is connected.",
                ),
                io.Int.Input(
                    "mask_feather",
                    default=12,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Soft Gaussian feather radius in pixels along the outer edge of face_mask.",
                ),
                io.Boolean.Input(
                    "anatomical_guides",
                    default=True,
                    label_on="Visible",
                    label_off="Hidden",
                    tooltip="Show toggleable minimal neck column cylinder and ear anchor markers.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    id="lit_color_pass",
                    display_name="lit_color_pass",
                    tooltip="Textured, 3D-lit face render batch.",
                ),
                io.Image.Output(
                    id="depth_pass",
                    display_name="depth_pass",
                    tooltip="Normalized grayscale Z-depth map batch.",
                ),
                io.Image.Output(
                    id="normal_pass",
                    display_name="normal_pass",
                    tooltip="RGB surface normal batch.",
                ),
                io.Mask.Output(
                    id="face_mask",
                    display_name="face_mask",
                    tooltip="Soft-feathered alpha silhouette mask of the 3D face.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        front_image: torch.Tensor,
        depth_weight: float,
        render_width: int,
        render_height: int,
        turntable_frames: int,
        mask_feather: int,
        anatomical_guides: bool,
        three_quarter_image: torch.Tensor | None = None,
        side_image: torch.Tensor | None = None,
        driving_video: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        # Push reference images (and optional driving video batch ID) to the browser viewport.
        prompt_server = getattr(PromptServer, "instance", None)
        node_id = getattr(cls.hidden, "unique_id", None)
        if prompt_server is not None and node_id is not None:
            driving_frames = _batch_to_data_urls(driving_video, max_side=512)
            driving_batch_id = cache_driving_batch(driving_frames) if driving_frames else None
            prompt_server.send_sync(
                "yedp_face3d_sync_refs",
                {
                    "node_id": str(node_id),
                    "front": _tensor_to_data_url(front_image),
                    "three_quarter": _tensor_to_data_url(three_quarter_image),
                    "side": _tensor_to_data_url(side_image),
                    "depth_weight": float(depth_weight),
                    "driving_batch_id": driving_batch_id,
                    "driving_frame_count": len(driving_frames),
                },
            )

        # Phases 3-4 will add animation playback and offscreen multi-pass WebGL capture.
        lit_pass, depth_pass, normal_pass, face_mask = build_phase1_placeholder_passes(
            front_image=front_image,
            driving_video=driving_video,
            depth_weight=depth_weight,
            render_width=render_width,
            render_height=render_height,
            turntable_frames=turntable_frames,
            mask_feather=mask_feather,
            anatomical_guides=anatomical_guides,
        )
        return io.NodeOutput(lit_pass, depth_pass, normal_pass, face_mask)
