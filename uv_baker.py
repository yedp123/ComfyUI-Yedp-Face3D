import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .face_math import (
    BROW_INDICES,
    EYE_CORNER_INDICES,
    EYE_RIM_FOLD_INDICES,
    LIP_CONTOUR_INDICES,
    MAX_FACE_MESH_VERTS,
    NASAL_BRIDGE_INDICES,
    NOSE_TIP_IDX,
    NOSE_TIP_NOSTRIL_INDICES,
    NOSTRIL_RIM_INDICES,
    _compute_face_normals,
    _estimate_effective_visibility_pose,
    canonicalize_view,
    compute_view_visibility_weights,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_CANONICAL_FACE_PATH = _PACKAGE_DIR / "assets" / "canonical_face.json"

VIEW_ORDER = ("front", "three_quarter", "three_quarter_2", "side", "side_2")
VIEW_SHORT_LABELS = {
    "front": "front",
    "three_quarter": "3q",
    "three_quarter_2": "3q2",
    "side": "side",
    "side_2": "side2",
}

# Debug source-map colors (RGB)
SOURCE_COLORS_RGB = {
    -1: (55, 65, 81),     # #374151 Uncovered / fallback
    0: (59, 130, 246),    # #3b82f6 front
    1: (34, 197, 94),     # #22c55e three_quarter (3/4 A)
    2: (6, 182, 212),     # #06b6d4 three_quarter_2 (3/4 B)
    3: (249, 115, 22),    # #f97316 side (Side A)
    4: (236, 72, 153),    # #ec4899 side_2 (Side B)
}

HAIR_RED_RGB = np.array([239, 68, 68], dtype=np.uint8)        # #ef4444
RECOVERED_GREEN_RGB = np.array([34, 197, 94], dtype=np.uint8) # #22c55e
UNCOVERED_GREY_RGB = np.array([55, 65, 81], dtype=np.uint8)   # #374151

# Split feature protection:
# 1) strict_front_lock locks fragile eyes, brows, lips, nostril rims, and narrow sagittal nose ridge/tip,
#    explicitly leaving nose side walls / alar slopes and inner-to-outer cheekbones unlocked (0.0).
# 2) hair_protect_mask shields facial features from SelfieMulticlass / outer-rim darkness rejection.
_NOSTRIL_OPENING_RIM_INDICES = NOSTRIL_RIM_INDICES - frozenset((48, 278, 115, 344))
_SAGITTAL_NOSE_RIDGE_INDICES = NASAL_BRIDGE_INDICES | frozenset((1, 4, 19, 94, 2))
STRICT_FRONT_LOCK_INDICES = (
    EYE_CORNER_INDICES
    | EYE_RIM_FOLD_INDICES
    | BROW_INDICES
    | LIP_CONTOUR_INDICES
    | _NOSTRIL_OPENING_RIM_INDICES
    | _SAGITTAL_NOSE_RIDGE_INDICES
)
HAIR_PROTECT_INDICES = (
    EYE_CORNER_INDICES
    | EYE_RIM_FOLD_INDICES
    | BROW_INDICES
    | LIP_CONTOUR_INDICES
    | NOSE_TIP_NOSTRIL_INDICES
    | NASAL_BRIDGE_INDICES
)

# Eye-socket fan topology so baked canonical textures populate EyeballRig iris/sclera regions cleanly
_RIGHT_EYE_RIM = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_LEFT_EYE_RIM = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
_EYE_SOCKET_TRIS = np.array(
    [(468, _RIGHT_EYE_RIM[k], _RIGHT_EYE_RIM[(k + 1) % len(_RIGHT_EYE_RIM)]) for k in range(len(_RIGHT_EYE_RIM))]
    + [(473, _LEFT_EYE_RIM[(k + 1) % len(_LEFT_EYE_RIM)], _LEFT_EYE_RIM[k]) for k in range(len(_LEFT_EYE_RIM))],
    dtype=np.int32,
)

_CANONICAL_UVS_CACHE: np.ndarray | None = None
_CANONICAL_TRIS_CACHE: np.ndarray | None = None
_ATLAS_PATCH_CACHE: dict[tuple[int, bytes], list[dict]] = {}


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep supporting both ascending (edge0 < edge1) and descending (edge0 > edge1) ramps."""
    diff = float(edge1 - edge0)
    if abs(diff) < 1e-6:
        return (x >= edge1).astype(np.float32) if diff >= 0.0 else (x <= edge1).astype(np.float32)
    t = np.clip((x - edge0) / diff, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def get_canonical_atlas() -> tuple[np.ndarray, np.ndarray]:
    """Load and cache MediaPipe's unwrapped canonical UV coordinates (478x2) and default 852 triangles."""
    global _CANONICAL_UVS_CACHE, _CANONICAL_TRIS_CACHE
    if _CANONICAL_UVS_CACHE is None or _CANONICAL_TRIS_CACHE is None:
        with open(_CANONICAL_FACE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        uvs = np.asarray(data["uv_front"], dtype=np.float32).reshape(-1, 2)
        tris = np.asarray(data["indices"], dtype=np.int32).reshape(-1, 3)
        _CANONICAL_UVS_CACHE = uvs
        _CANONICAL_TRIS_CACHE = tris
    return _CANONICAL_UVS_CACHE, _CANONICAL_TRIS_CACHE


def validate_uv_atlas(uv_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Verify mathematically that all face triangles have non-degenerate area and zero UV fold edges,
    returning CCW-oriented UV triangles (`area_uv > 0` for all triangles).
    """
    tri_arr = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    face_mask = np.all((tri_arr >= 0) & (tri_arr < MAX_FACE_MESH_VERTS), axis=1)
    tri_arr = tri_arr[face_mask].copy()
    if len(tri_arr) == 0:
        raise ValueError("No valid face triangles (0..467) provided for UV baking.")

    uv_face = uv_coords[:MAX_FACE_MESH_VERTS]
    if not np.all(np.isfinite(uv_face)):
        raise ValueError("Canonical UV atlas contains non-finite coordinates.")

    a = uv_face[tri_arr[:, 0]]
    b = uv_face[tri_arr[:, 1]]
    c = uv_face[tri_arr[:, 2]]
    signed_area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    degen_count = int(np.sum(np.abs(signed_area) < 1e-7))
    if degen_count > 0:
        raise ValueError(f"UV atlas has {degen_count} degenerate near-zero area triangles.")

    edges: dict[tuple[int, int], list[int]] = {}
    for ia, ib, ic in tri_arr:
        for u, v, w in ((int(ia), int(ib), int(ic)), (int(ib), int(ic), int(ia)), (int(ic), int(ia), int(ib))):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append(w)

    fold_count = 0
    for (e0, e1), opp in edges.items():
        if len(opp) != 2:
            continue
        p0, p1 = uv_face[e0], uv_face[e1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        w1, w2 = uv_face[opp[0]], uv_face[opp[1]]
        s1 = dx * float(w1[1] - p0[1]) - dy * float(w1[0] - p0[0])
        s2 = dx * float(w2[1] - p0[1]) - dy * float(w2[0] - p0[0])
        if s1 * s2 >= 0.0:
            fold_count += 1
    if fold_count > 0:
        raise ValueError(f"Canonical UV atlas contains {fold_count} folded interior edges.")

    flip_mask = signed_area < 0.0
    if np.any(flip_mask):
        tri_arr[flip_mask, 1], tri_arr[flip_mask, 2] = tri_arr[flip_mask, 2], tri_arr[flip_mask, 1]
    return tri_arr


def _build_vertex_mask(n_verts: int, indices: frozenset[int], include_iris: bool = True) -> np.ndarray:
    """Build per-vertex [0.0 / 1.0] mask on canonical vertices."""
    mask = np.zeros(n_verts, dtype=np.float32)
    for idx in indices:
        if 0 <= idx < n_verts:
            mask[idx] = 1.0
    if include_iris and n_verts > 468:
        mask[468:n_verts] = 1.0
    return mask


def _build_uv_patch_plan(
    uv_coords: np.ndarray,
    tri_arr: np.ndarray,
    atlas_size: int,
) -> list[dict]:
    """Precompute per-triangle target UV bounding boxes, affine coords, and barycentric weights."""
    key = (int(atlas_size), tri_arr.tobytes())
    cached = _ATLAS_PATCH_CACHE.get(key)
    if cached is not None:
        return cached

    uv_px = np.empty_like(uv_coords, dtype=np.float32)
    uv_px[:, 0] = uv_coords[:, 0] * float(atlas_size) - 0.5
    uv_px[:, 1] = (1.0 - uv_coords[:, 1]) * float(atlas_size) - 0.5

    patches: list[dict] = []
    for a, b, c in tri_arr:
        ai, bi, ci = int(a), int(b), int(c)
        p0, p1, p2 = uv_px[ai], uv_px[bi], uv_px[ci]
        x0 = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x1 = min(atlas_size, int(np.ceil(max(p0[0], p1[0], p2[0]))) + 1)
        y0 = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y1 = min(atlas_size, int(np.ceil(max(p0[1], p1[1], p2[1]))) + 1)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue

        v0 = p1 - p0
        v1 = p2 - p0
        denom = float(v0[0] * v1[1] - v1[0] * v0[1])
        area_uv_px = 0.5 * abs(denom)
        if area_uv_px < 1e-6:
            continue

        inv_d = 1.0 / denom
        xs = np.arange(x0, x1, dtype=np.float32) - p0[0]
        ys = np.arange(y0, y1, dtype=np.float32) - p0[1]
        gx, gy = np.meshgrid(xs, ys)
        w1 = (gx * v1[1] - gy * v1[0]) * inv_d
        w2 = (v0[0] * gy - v0[1] * gx) * inv_d
        w0 = 1.0 - w1 - w2
        mask = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if not np.any(mask):
            continue

        b0 = np.clip(w0[mask], 0.0, 1.0).astype(np.float32)
        b1 = np.clip(w1[mask], 0.0, 1.0).astype(np.float32)
        b2 = np.clip(w2[mask], 0.0, 1.0).astype(np.float32)
        b_sum = np.maximum(b0 + b1 + b2, 1e-6)
        b0 /= b_sum
        b1 /= b_sum
        b2 /= b_sum

        d_tri = np.array(
            [
                [p0[0] - x0, p0[1] - y0],
                [p1[0] - x0, p1[1] - y0],
                [p2[0] - x0, p2[1] - y0],
            ],
            dtype=np.float32,
        )
        patches.append(
            {
                "a": ai,
                "b": bi,
                "c": ci,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "bw": bw,
                "bh": bh,
                "d_tri": d_tri,
                "mask": mask,
                "b0": b0,
                "b1": b1,
                "b2": b2,
                "area_uv_px": area_uv_px,
            }
        )

    if len(_ATLAS_PATCH_CACHE) < 8:
        _ATLAS_PATCH_CACHE[key] = patches
    return patches


def _rasterize_canonical_fields(
    atlas_size: int,
    face_patches: list[dict],
    eye_patches: list[dict],
    front_pts: np.ndarray,
    oriented_face_tris: np.ndarray,
    n_uv_verts: int,
) -> dict[str, np.ndarray]:
    """Rasterize canonical 3D surface normals, signed X position, and feature locks into UV space."""
    n_face = min(len(front_pts), MAX_FACE_MESH_VERTS)
    face_normals = _compute_face_normals(front_pts[:n_face], oriented_face_tris)
    if n_face > NOSE_TIP_IDX and face_normals[NOSE_TIP_IDX, 2] < 0.0:
        face_normals = -face_normals

    vert_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n_uv_verts, 1))
    vert_normals[:n_face] = face_normals

    half_width = max(float(np.max(np.abs(front_pts[:n_face, 0]))), 1e-5)
    x_signed = np.zeros(n_uv_verts, dtype=np.float32)
    x_signed[: min(n_uv_verts, len(front_pts))] = (
        front_pts[: min(n_uv_verts, len(front_pts)), 0] / half_width
    ).astype(np.float32)

    strict_lock_v = _build_vertex_mask(n_uv_verts, STRICT_FRONT_LOCK_INDICES, include_iris=True)
    hair_protect_v = _build_vertex_mask(n_uv_verts, HAIR_PROTECT_INDICES, include_iris=True)

    face_covered = np.zeros((atlas_size, atlas_size), dtype=bool)
    eye_covered = np.zeros((atlas_size, atlas_size), dtype=bool)
    uv_normal = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)
    uv_normal[:, :, 2] = 1.0
    uv_x_signed = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    uv_strict_lock = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    uv_hair_protect = np.zeros((atlas_size, atlas_size), dtype=np.float32)

    for p in face_patches:
        a, b, c = p["a"], p["b"], p["c"]
        y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
        mask = p["mask"]
        b0, b1, b2 = p["b0"], p["b1"], p["b2"]

        face_covered[y0:y1, x0:x1][mask] = True
        uv_x_signed[y0:y1, x0:x1][mask] = (
            b0 * x_signed[a] + b1 * x_signed[b] + b2 * x_signed[c]
        ).astype(np.float32)
        uv_strict_lock[y0:y1, x0:x1][mask] = (
            b0 * strict_lock_v[a] + b1 * strict_lock_v[b] + b2 * strict_lock_v[c]
        ).astype(np.float32)
        uv_hair_protect[y0:y1, x0:x1][mask] = (
            b0 * hair_protect_v[a] + b1 * hair_protect_v[b] + b2 * hair_protect_v[c]
        ).astype(np.float32)
        n_interp = b0[:, None] * vert_normals[a] + b1[:, None] * vert_normals[b] + b2[:, None] * vert_normals[c]
        uv_normal[y0:y1, x0:x1][mask] = n_interp

    # Re-normalize interpolated unit normals strictly across face pixels
    n_len = np.linalg.norm(uv_normal, axis=2, keepdims=True)
    uv_normal = (uv_normal / np.maximum(n_len, 1e-6)).astype(np.float32)

    for p in eye_patches:
        y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
        mask = p["mask"]
        eye_covered[y0:y1, x0:x1][mask] = True
        uv_strict_lock[y0:y1, x0:x1][mask] = 1.0
        uv_hair_protect[y0:y1, x0:x1][mask] = 1.0

    return {
        "face_covered": face_covered,
        "eye_covered": eye_covered,
        "uv_normal": uv_normal,
        "uv_x_signed": uv_x_signed,
        "uv_x_abs": np.abs(uv_x_signed).astype(np.float32),
        "uv_strict_lock": uv_strict_lock,
        "uv_hair_protect": uv_hair_protect,
    }


def _decode_rgb_image(view_dict: dict) -> np.ndarray:
    """Decode view image from `image_rgb`, disk `image_path`, or `image_data_url` as uint8 RGB."""
    raw_rgb = view_dict.get("image_rgb")
    if isinstance(raw_rgb, np.ndarray) and raw_rgb.ndim == 3 and raw_rgb.shape[2] >= 3:
        return np.ascontiguousarray(raw_rgb[:, :, :3], dtype=np.uint8)

    img_path = view_dict.get("image_path")
    if img_path:
        raw_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    data_url = view_dict.get("image_data_url")
    if data_url and isinstance(data_url, str):
        b64 = data_url.split(",", 1)[-1]
        raw_bytes = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError("View is missing readable image pixels (image_path or image_data_url).")


def _decode_seg_mask(view_dict: dict) -> np.ndarray | None:
    """Decode compact 256x256 RGB segmentation mask (R=face_skin, G=hair, B=non_skin)."""
    raw_seg = view_dict.get("seg_mask_rgb")
    if isinstance(raw_seg, np.ndarray) and raw_seg.ndim == 3 and raw_seg.shape[2] >= 3:
        return np.ascontiguousarray(raw_seg[:, :, :3], dtype=np.uint8)

    seg_url = view_dict.get("seg_mask_png")
    if not seg_url or not isinstance(seg_url, str):
        return None
    b64 = seg_url.split(",", 1)[-1]
    raw_bytes = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_png_data_url(rgb_img: np.ndarray) -> str:
    """Encode a uint8 RGB NumPy array into a base64 PNG data URL."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG output.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _dilate_uv_bleed(rgb_img: np.ndarray, covered_mask: np.ndarray, radius_px: int = 6) -> np.ndarray:
    """Extend valid UV edge texels outward by `radius_px` via Voronoi distance transform."""
    if not np.any(covered_mask):
        return rgb_img
    inv_mask = (~covered_mask).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    bleed = (~covered_mask) & (dist <= float(radius_px))
    if not np.any(bleed):
        return rgb_img
    vy, vx = np.where(covered_mask)
    nearest_idx = labels[bleed] - 1
    out = rgb_img.copy()
    out[bleed] = rgb_img[vy[nearest_idx], vx[nearest_idx]]
    return out


def _compute_view_yaw_signed(view_dict: dict) -> float:
    """Estimate signed effective head yaw in degrees using face_math Phase 1.5 estimator."""
    _, _, pose = canonicalize_view(
        view_dict["landmarks"],
        view_dict.get("matrix"),
        int(view_dict.get("width", 512)),
        int(view_dict.get("height", 512)),
    )
    vis_pose = _estimate_effective_visibility_pose(view_dict["landmarks"], pose)
    return float(vis_pose["yaw_eff_deg"])


def _masked_gaussian_blur(field: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized masked Gaussian blur keeping boundary energy intact inside `valid_mask`."""
    if sigma <= 0.1 or not np.any(valid_mask):
        return field
    valid_f = valid_mask.astype(np.float32)
    num = cv2.GaussianBlur(field * valid_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    den = cv2.GaussianBlur(valid_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    return np.where(valid_mask, num / np.maximum(den, 1e-5), 0.0).astype(np.float32)


def _cleanup_small_islands(
    winner_id: np.ndarray,
    face_covered: np.ndarray,
    active_indices: list[int],
    min_island_px: int,
) -> np.ndarray:
    """Absorb isolated connected components smaller than `min_island_px` into their majority neighbor view."""
    if min_island_px <= 1 or len(active_indices) <= 1:
        return winner_id
    out = winner_id.copy()
    h, w = out.shape
    kernel = np.ones((3, 3), dtype=np.uint8)

    for v_idx in active_indices:
        bin_mask = ((out == v_idx) & face_covered).astype(np.uint8)
        if not np.any(bin_mask):
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels <= 1:
            continue
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area >= min_island_px:
                continue
            bx = int(stats[comp_id, cv2.CC_STAT_LEFT])
            by = int(stats[comp_id, cv2.CC_STAT_TOP])
            bw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            x0, x1 = max(0, bx - 1), min(w, bx + bw + 1)
            y0, y1 = max(0, by - 1), min(h, by + bh + 1)

            comp_sub = (labels[y0:y1, x0:x1] == comp_id).astype(np.uint8)
            dil_sub = cv2.dilate(comp_sub, kernel, iterations=1) > 0
            border_sub = dil_sub & (comp_sub == 0) & face_covered[y0:y1, x0:x1]
            nbr_labels = out[y0:y1, x0:x1][border_sub]
            nbr_labels = nbr_labels[(nbr_labels >= 0) & (nbr_labels != v_idx)]
            if len(nbr_labels) == 0:
                continue
            counts = np.bincount(nbr_labels.astype(np.int32))
            majority_label = int(np.argmax(counts))
            out[y0:y1, x0:x1][comp_sub > 0] = majority_label
    return out


def _trimmed_lab_stats(vals: np.ndarray) -> tuple[float, float]:
    """Compute 5th-to-95th percentile trimmed mean and standard deviation."""
    if len(vals) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [5.0, 95.0])
    trimmed = vals[(vals >= lo) & (vals <= hi)]
    if len(trimmed) < 16:
        trimmed = vals
    return float(np.mean(trimmed)), float(np.std(trimmed))


def _rgb_to_lab_f32(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to CIE Lab float32 (L in [0, 100], a/b in [-127, 127])."""
    return cv2.cvtColor(rgb_u8.astype(np.float32) * np.float32(1.0 / 255.0), cv2.COLOR_RGB2LAB)


def _median_cheek_luminance(
    lab_l: np.ndarray,
    face_valid: np.ndarray,
    vis: np.ndarray,
    hair_raw: np.ndarray,
    uv_x_abs: np.ndarray,
    uv_hair_protect: np.ndarray,
) -> float | None:
    """Measure clean mid-cheek median CIE L* luminance across abs(x_signed_norm) in [0.25, 0.65]."""
    clean_cheek = (
        face_valid
        & (vis >= 0.15)
        & (uv_x_abs >= 0.25)
        & (uv_x_abs <= 0.65)
        & (hair_raw < 0.10)
        & (uv_hair_protect < 0.10)
    )
    if int(np.sum(clean_cheek)) >= 100:
        return float(np.median(lab_l[clean_cheek]))

    fallback = face_valid & (vis >= 0.15) & (hair_raw < 0.10) & (uv_hair_protect < 0.10)
    if int(np.sum(fallback)) >= 50:
        return float(np.median(lab_l[fallback]))
    return None


def _harmonize_view_lab(
    front_lab: np.ndarray,
    front_valid: np.ndarray,
    front_vis: np.ndarray,
    front_hair: np.ndarray,
    front_non_skin: np.ndarray,
    aux_rgb: np.ndarray,
    aux_valid: np.ndarray,
    aux_vis: np.ndarray,
    aux_hair: np.ndarray,
    aux_non_skin: np.ndarray,
    uv_protect: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """
    Match auxiliary view RGB into front's CIE Lab skin distribution using strictly clean,
    non-feature overlapping skin texels (`M_overlap >= 400`).
    """
    overlap = (
        front_valid
        & aux_valid
        & (front_vis > 0.25)
        & (aux_vis > 0.25)
        & (front_hair < 0.15)
        & (aux_hair < 0.15)
        & (front_non_skin < 0.15)
        & (aux_non_skin < 0.15)
        & (uv_protect < 0.25)
    )
    if int(np.sum(overlap)) < 400:
        return aux_rgb, None

    aux_lab = _rgb_to_lab_f32(aux_rgb)
    f_pts = front_lab[overlap]
    v_pts = aux_lab[overlap]

    mu_f_l, _ = _trimmed_lab_stats(f_pts[:, 0])
    mu_v_l, _ = _trimmed_lab_stats(v_pts[:, 0])
    dL = mu_f_l - mu_v_l

    matched_lab = aux_lab.copy()
    for ch in range(3):
        mu_f, sd_f = _trimmed_lab_stats(f_pts[:, ch])
        mu_v, sd_v = _trimmed_lab_stats(v_pts[:, ch])
        scale = float(np.clip(sd_f / max(sd_v, 1.0), 0.75, 1.30))
        matched_lab[:, :, ch] = (aux_lab[:, :, ch] - mu_v) * scale + mu_f

    matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0.0, 100.0)
    matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1], -127.0, 127.0)
    matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2], -127.0, 127.0)
    matched_rgb_f = np.clip(cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    matched_rgb = np.round(matched_rgb_f * 255.0).astype(np.uint8)
    return matched_rgb, round(dL, 1)


def _compute_clean_push_pull_field(
    rgb_img: np.ndarray,
    clean_src_mask: np.ndarray,
    atlas_size: int,
) -> np.ndarray | None:
    """
    Build normalized multi-scale push-pull skin field from `clean_src_mask` only
    (downsampling to ~32x32 and pulling back to full `atlas_size`), zero background/hair bleed.
    """
    if not np.any(clean_src_mask):
        return None

    clean_f = clean_src_mask.astype(np.float32)
    clean_mean = rgb_img[clean_src_mask].astype(np.float32).mean(axis=0)
    p_cur = rgb_img.astype(np.float32) * clean_f[:, :, None]
    w_cur = clean_f

    push_p: list[np.ndarray] = [p_cur]
    push_w: list[np.ndarray] = [w_cur]
    while min(p_cur.shape[0], p_cur.shape[1]) > 32:
        p_cur = cv2.pyrDown(p_cur)
        w_cur = cv2.pyrDown(w_cur)
        push_p.append(p_cur)
        push_w.append(w_cur)

    # Normalize coarsest (~32x32) level strictly from clean weights
    w_coarse = push_w[-1]
    p_coarse = push_p[-1]
    pull_rgb = np.where(
        w_coarse[:, :, None] > 1e-6,
        p_coarse / np.maximum(w_coarse[:, :, None], 1e-6),
        clean_mean[None, None, :],
    ).astype(np.float32)

    for k in range(len(push_p) - 2, -1, -1):
        p_lvl = push_p[k]
        w_lvl = push_w[k]
        up = cv2.pyrUp(pull_rgb, dstsize=(p_lvl.shape[1], p_lvl.shape[0]))
        fine_rgb = np.where(
            w_lvl[:, :, None] > 1e-6,
            p_lvl / np.maximum(w_lvl[:, :, None], 1e-6),
            up,
        ).astype(np.float32)
        c_lvl = np.clip(w_lvl, 0.0, 1.0)[:, :, None]
        pull_rgb = c_lvl * fine_rgb + (1.0 - c_lvl) * up

    return np.clip(pull_rgb, 0.0, 255.0).astype(np.float32)


def _apply_push_pull_fill(
    rgb_f32: np.ndarray,
    pull_field: np.ndarray | None,
    fill_mask: np.ndarray,
    face_covered: np.ndarray,
    uv_hair_protect: np.ndarray,
    atlas_size: int,
) -> np.ndarray:
    """Composite `pull_field` strictly over `fill_mask` with a sigma=4px feathered edge."""
    if pull_field is None or not np.any(fill_mask):
        return rgb_f32

    scale = float(atlas_size) / 1024.0
    sigma = max(0.8, 4.0 * scale)
    alpha = cv2.GaussianBlur(
        fill_mask.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    alpha = np.where(fill_mask, 1.0, alpha)
    alpha = np.where(face_covered & (uv_hair_protect < 0.15), np.clip(alpha, 0.0, 1.0), 0.0)[:, :, None]
    return np.clip(rgb_f32 * (1.0 - alpha) + pull_field * alpha, 0.0, 255.0).astype(np.float32)


def _blend_laplacian_pyramids(
    I_base_pad_f32: np.ndarray,
    winning_views: list[tuple[str, dict]],
    winner_id: np.ndarray,
    any_covered: np.ndarray,
    atlas_size: int,
) -> np.ndarray:
    """
    5-level frequency-split Laplacian pyramid blend across seam borders:
    - Level 0 (1024px): tight sigma=2px weight transition (zero ghosting/double details)
    - Level 1 (512px): tight-medium sigma=3px weight transition
    - Levels 2..4 (256px..64px): smooth Gaussian weight pyramid with extra sigma=2.5px blur
    """
    scale = float(atlas_size) / 1024.0
    num_levels = 5
    while num_levels > 2 and (atlas_size >> (num_levels - 1)) < 16:
        num_levels -= 1

    feather_px = max(1.0, 3.0 * scale)
    sigma_l0 = max(0.8, 2.0 * scale)
    sigma_l1 = max(0.8, 3.0 * scale)
    sigma_coarse = max(0.8, 2.5 * scale)

    lap_accum: list[np.ndarray] | None = None
    weight_sums: list[np.ndarray] | None = None
    view_pyramids: list[tuple[list[np.ndarray], list[np.ndarray]]] = []

    for _, v_buf in winning_views:
        v_clean_valid = (
            (v_buf["face_valid"] | v_buf["eye_valid"])
            & (v_buf["vis"] >= 0.15)
            & (~v_buf["is_bad"])
        )
        if np.any(v_clean_valid):
            dist_in = cv2.distanceTransform(v_clean_valid.astype(np.uint8), cv2.DIST_L2, 3)
            a_v = np.clip(dist_in / feather_px, 0.0, 1.0)[:, :, None].astype(np.float32)
            P_v = v_buf["rgb"].astype(np.float32) * a_v + I_base_pad_f32 * (1.0 - a_v)
        else:
            P_v = I_base_pad_f32

        # Build Gaussian & Laplacian stacks for P_v
        g_stack = [P_v]
        for _ in range(num_levels - 1):
            g_stack.append(cv2.pyrDown(g_stack[-1]))
        l_stack: list[np.ndarray] = []
        for k in range(num_levels - 1):
            up = cv2.pyrUp(g_stack[k + 1], dstsize=(g_stack[k].shape[1], g_stack[k].shape[0]))
            l_stack.append(g_stack[k] - up)
        l_stack.append(g_stack[-1])

        # Build frequency-split weight stack for ownership mask M_v
        M_v = ((winner_id == v_buf["idx"]) & any_covered).astype(np.float32)
        m_pyr = [M_v]
        for _ in range(num_levels - 1):
            m_pyr.append(cv2.pyrDown(m_pyr[-1]))

        w_raw: list[np.ndarray] = []
        for k in range(num_levels):
            if k == 0:
                sig = sigma_l0
            elif k == 1:
                sig = sigma_l1
            else:
                sig = sigma_coarse
            w_raw.append(cv2.GaussianBlur(m_pyr[k], (0, 0), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_REPLICATE))

        if weight_sums is None:
            weight_sums = [w.copy() for w in w_raw]
        else:
            for k in range(num_levels):
                weight_sums[k] += w_raw[k]

        view_pyramids.append((l_stack, w_raw))

    assert weight_sums is not None
    lap_accum = [np.zeros_like(view_pyramids[0][0][k], dtype=np.float32) for k in range(num_levels)]
    for l_stack, w_raw in view_pyramids:
        for k in range(num_levels):
            w_norm = w_raw[k] / np.maximum(weight_sums[k], 1e-6)
            lap_accum[k] += w_norm[:, :, None] * l_stack[k]

    recon = lap_accum[-1]
    for k in range(num_levels - 2, -1, -1):
        recon = cv2.pyrUp(recon, dstsize=(lap_accum[k].shape[1], lap_accum[k].shape[0])) + lap_accum[k]

    return np.clip(recon, 0.0, 255.0).astype(np.float32)


def bake_uv_texture(
    views_by_key: dict[str, dict],
    triangles: list[int] | np.ndarray,
    atlas_size: int = 1024,
    debug_size: int = 512,
    enable_color_match: bool = True,
    enable_seam_blend: bool = True,
) -> dict:
    """
    Bake a multi-view UV texture in MediaPipe's unwrapped canonical UV layout using
    3D surface-normal camera facing, far-side guard, split feature protection, hair halo +
    outer-rim dark shadow push-pull cleanup, overlap CIE Lab matching, and optional 5-level
    seam-only Laplacian pyramid blending.
    """
    t_start = time.perf_counter()
    atlas_size = int(atlas_size)
    debug_size = int(debug_size)
    enable_color_match = bool(enable_color_match)
    enable_seam_blend = bool(enable_seam_blend)
    scale = float(atlas_size) / 1024.0

    active_keys = [k for k in VIEW_ORDER if views_by_key.get(k) and views_by_key[k].get("landmarks")]
    if "front" not in active_keys:
        raise ValueError("Missing required 'front' view for UV baking.")

    canonical_uvs, default_tris = get_canonical_atlas()
    tri_input = np.asarray(triangles, dtype=np.int32).reshape(-1, 3) if triangles is not None else default_tris
    oriented_face_tris = validate_uv_atlas(canonical_uvs, tri_input)

    vis_info = compute_view_visibility_weights(views_by_key, oriented_face_tris.ravel().tolist())
    vis_weights = vis_info["vis_weights"]
    front_pts = vis_info["front_pts"]

    n_uv_verts = len(canonical_uvs)
    face_patches = _build_uv_patch_plan(canonical_uvs, oriented_face_tris, atlas_size)
    eye_patches = _build_uv_patch_plan(canonical_uvs, _EYE_SOCKET_TRIS, atlas_size)

    canon_fields = _rasterize_canonical_fields(
        atlas_size, face_patches, eye_patches, front_pts, oriented_face_tris, n_uv_verts
    )
    face_covered = canon_fields["face_covered"]
    eye_covered = canon_fields["eye_covered"]
    uv_normal = canon_fields["uv_normal"]
    uv_x_signed = canon_fields["uv_x_signed"]
    uv_x_abs = canon_fields["uv_x_abs"]
    uv_strict_lock = canon_fields["uv_strict_lock"]
    uv_hair_protect = canon_fields["uv_hair_protect"]

    inv255 = np.float32(1.0 / 255.0)
    view_to_idx = {k: i for i, k in enumerate(VIEW_ORDER)}

    hair_rad = max(1, int(round(8.0 * scale)))
    hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hair_rad + 1, 2 * hair_rad + 1))
    hair_blur_sigma = max(0.5, 4.0 * scale)
    score_blur_sigma = max(1.0, 10.0 * scale)
    min_island_px = max(16, int(round(250.0 * scale * scale)))

    # Step A: Project every active view onto UV space
    view_buffers: dict[str, dict] = {}
    for view_key in active_keys:
        view_dict = views_by_key[view_key]
        rgb_img = _decode_rgb_image(view_dict)
        h_src, w_src = rgb_img.shape[:2]
        seg_img = _decode_seg_mask(view_dict)
        yaw_eff_deg = _compute_view_yaw_signed(view_dict)

        if view_key == "front":
            view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            yaw_rad = float(np.radians(yaw_eff_deg))
            view_dir = np.array([-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)], dtype=np.float32)

        lms = view_dict["landmarks"]
        n_lm = len(lms)
        src_xy_rgb = np.empty((n_lm, 2), dtype=np.float32)
        src_xy_yup = np.empty((n_lm, 2), dtype=np.float32)
        for i, lm in enumerate(lms):
            lx = float(lm["x"])
            ly = float(lm["y"])
            src_xy_rgb[i, 0] = lx * float(w_src) - 0.5
            src_xy_rgb[i, 1] = ly * float(h_src) - 0.5
            src_xy_yup[i, 0] = lx * float(w_src)
            src_xy_yup[i, 1] = (1.0 - ly) * float(h_src)

        if seg_img is not None:
            h_seg, w_seg = seg_img.shape[:2]
            src_xy_seg = np.empty((n_lm, 2), dtype=np.float32)
            for i, lm in enumerate(lms):
                src_xy_seg[i, 0] = float(lm["x"]) * float(w_seg) - 0.5
                src_xy_seg[i, 1] = float(lm["y"]) * float(h_seg) - 0.5
        else:
            src_xy_seg = None

        min_area_2d = 1e-7 * float(w_src * h_src)
        vis_v = vis_weights[view_key]
        n_vis = len(vis_v)

        # Smooth C0 per-vertex resolution factor clamped to [0.75, 1.15]
        v_res_num = np.zeros(n_uv_verts, dtype=np.float32)
        v_res_den = np.zeros(n_uv_verts, dtype=np.float32)
        valid_face_tris: list[dict] = []
        for p in face_patches:
            a, b, c = p["a"], p["b"], p["c"]
            if a >= n_lm or b >= n_lm or c >= n_lm or a >= n_vis or b >= n_vis or c >= n_vis:
                continue
            va, vb, vc = float(vis_v[a]), float(vis_v[b]), float(vis_v[c])
            if va < 0.05 and vb < 0.05 and vc < 0.05:
                continue
            q0, q1, q2 = src_xy_yup[a], src_xy_yup[b], src_xy_yup[c]
            area_2d = 0.5 * float((q1[0] - q0[0]) * (q2[1] - q0[1]) - (q1[1] - q0[1]) * (q2[0] - q0[0]))
            if area_2d <= min_area_2d:
                continue
            tri_res = float(np.clip(np.sqrt(area_2d / max(p["area_uv_px"], 1e-6)), 0.75, 1.15))
            v_res_num[a] += tri_res * area_2d
            v_res_num[b] += tri_res * area_2d
            v_res_num[c] += tri_res * area_2d
            v_res_den[a] += area_2d
            v_res_den[b] += area_2d
            v_res_den[c] += area_2d
            valid_face_tris.append(p)

        vert_res = np.where(v_res_den > 1e-6, v_res_num / np.maximum(v_res_den, 1e-6), 0.95).astype(np.float32)
        vert_res = np.clip(vert_res, 0.75, 1.15)

        v_rgb = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
        v_face_valid = np.zeros((atlas_size, atlas_size), dtype=bool)
        v_eye_valid = np.zeros((atlas_size, atlas_size), dtype=bool)
        v_vis = np.zeros((atlas_size, atlas_size), dtype=np.float32)
        v_res = np.zeros((atlas_size, atlas_size), dtype=np.float32)
        v_hair = np.zeros((atlas_size, atlas_size), dtype=np.float32)
        v_non_skin = np.zeros((atlas_size, atlas_size), dtype=np.float32)

        for p in valid_face_tris:
            a, b, c = p["a"], p["b"], p["c"]
            y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
            mask = p["mask"]

            s_tri_rgb = np.array([src_xy_rgb[a], src_xy_rgb[b], src_xy_rgb[c]], dtype=np.float32)
            M_rgb = cv2.getAffineTransform(s_tri_rgb, p["d_tri"])
            warped_rgb = cv2.warpAffine(
                rgb_img, M_rgb, (p["bw"], p["bh"]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            v_rgb[y0:y1, x0:x1][mask] = warped_rgb[mask]
            v_face_valid[y0:y1, x0:x1][mask] = True
            v_vis[y0:y1, x0:x1][mask] = (
                p["b0"] * float(vis_v[a]) + p["b1"] * float(vis_v[b]) + p["b2"] * float(vis_v[c])
            )
            v_res[y0:y1, x0:x1][mask] = (
                p["b0"] * float(vert_res[a]) + p["b1"] * float(vert_res[b]) + p["b2"] * float(vert_res[c])
            )

            if seg_img is not None and src_xy_seg is not None:
                s_tri_seg = np.array([src_xy_seg[a], src_xy_seg[b], src_xy_seg[c]], dtype=np.float32)
                M_seg = cv2.getAffineTransform(s_tri_seg, p["d_tri"])
                warped_seg = cv2.warpAffine(
                    seg_img, M_seg, (p["bw"], p["bh"]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )[mask]
                v_hair[y0:y1, x0:x1][mask] = warped_seg[:, 1].astype(np.float32) * inv255
                v_non_skin[y0:y1, x0:x1][mask] = warped_seg[:, 2].astype(np.float32) * inv255

        for p in eye_patches:
            a, b, c = p["a"], p["b"], p["c"]
            if a >= n_lm or b >= n_lm or c >= n_lm or a >= n_vis or b >= n_vis or c >= n_vis:
                continue
            va, vb, vc = float(vis_v[a]), float(vis_v[b]), float(vis_v[c])
            if va < 0.05 and vb < 0.05 and vc < 0.05:
                continue
            q0, q1, q2 = src_xy_yup[a], src_xy_yup[b], src_xy_yup[c]
            area_2d = 0.5 * float((q1[0] - q0[0]) * (q2[1] - q0[1]) - (q1[1] - q0[1]) * (q2[0] - q0[0]))
            if area_2d <= min_area_2d:
                continue
            y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
            mask = p["mask"]
            s_tri_rgb = np.array([src_xy_rgb[a], src_xy_rgb[b], src_xy_rgb[c]], dtype=np.float32)
            M_rgb = cv2.getAffineTransform(s_tri_rgb, p["d_tri"])
            warped_rgb = cv2.warpAffine(
                rgb_img, M_rgb, (p["bw"], p["bh"]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            v_rgb[y0:y1, x0:x1][mask] = warped_rgb[mask]
            v_eye_valid[y0:y1, x0:x1][mask] = True
            v_vis[y0:y1, x0:x1][mask] = p["b0"] * va + p["b1"] * vb + p["b2"] * vc
            v_res[y0:y1, x0:x1][mask] = float(np.clip(np.sqrt(area_2d / max(p["area_uv_px"], 1e-6)), 0.75, 1.15))

        if seg_img is not None and np.any(v_face_valid):
            hair_dil = cv2.dilate(v_hair, hair_kernel)
            hair_blur = cv2.GaussianBlur(
                hair_dil, (0, 0), sigmaX=hair_blur_sigma, sigmaY=hair_blur_sigma, borderType=cv2.BORDER_CONSTANT
            )
            v_hair_halo = np.where(v_face_valid, np.maximum(v_hair, hair_blur), 0.0).astype(np.float32)
        else:
            v_hair_halo = v_hair

        eff_hair = v_hair_halo * (1.0 - 0.8 * uv_hair_protect)

        facing = np.clip(
            uv_normal[:, :, 0] * float(view_dir[0])
            + uv_normal[:, :, 1] * float(view_dir[1])
            + uv_normal[:, :, 2] * float(view_dir[2]),
            0.0,
            1.0,
        ).astype(np.float32)

        if view_key != "front" and abs(yaw_eff_deg) >= 5.0:
            yaw_sign = 1.0 if yaw_eff_deg > 0.0 else -1.0
            far_side = uv_x_signed * yaw_sign
            far_guard = (1.0 - 0.95 * _smoothstep(0.03, 0.18, far_side)).astype(np.float32)
        else:
            far_guard = np.ones((atlas_size, atlas_size), dtype=np.float32)

        view_buffers[view_key] = {
            "idx": view_to_idx[view_key],
            "rgb": v_rgb,
            "face_valid": v_face_valid,
            "eye_valid": v_eye_valid,
            "vis": v_vis,
            "res": v_res,
            "hair_raw": v_hair,
            "eff_hair": eff_hair,
            "non_skin": v_non_skin,
            "facing": facing,
            "far_guard": far_guard,
            "has_seg": seg_img is not None,
        }

    # Step B: Overlap-Only CIE Lab Color Harmonization
    f_buf = view_buffers["front"]
    front_lab = _rgb_to_lab_f32(f_buf["rgb"])
    front_l_med = _median_cheek_luminance(
        front_lab[:, :, 0], f_buf["face_valid"], f_buf["vis"], f_buf["hair_raw"], uv_x_abs, uv_hair_protect
    )
    color_shifts_dl: dict[str, float | None] = {}
    if enable_color_match and len(active_keys) > 1:
        for view_key in active_keys:
            if view_key == "front":
                continue
            v_buf = view_buffers[view_key]
            matched_rgb, dL = _harmonize_view_lab(
                front_lab=front_lab,
                front_valid=f_buf["face_valid"],
                front_vis=f_buf["vis"],
                front_hair=f_buf["hair_raw"],
                front_non_skin=f_buf["non_skin"],
                aux_rgb=v_buf["rgb"],
                aux_valid=v_buf["face_valid"],
                aux_vis=v_buf["vis"],
                aux_hair=v_buf["hair_raw"],
                aux_non_skin=v_buf["non_skin"],
                uv_protect=uv_hair_protect,
            )
            v_buf["rgb"] = matched_rgb
            color_shifts_dl[view_key] = dL

    # Evaluate outer-rim dark hair shadow & final normal-driven view scores
    for view_key in active_keys:
        v_buf = view_buffers[view_key]
        v_lab_l = front_lab[:, :, 0] if view_key == "front" else _rgb_to_lab_f32(v_buf["rgb"])[:, :, 0]
        if view_key == "front" or enable_color_match:
            l_ref = front_l_med
        else:
            l_ref = _median_cheek_luminance(
                v_lab_l, v_buf["face_valid"], v_buf["vis"], v_buf["hair_raw"], uv_x_abs, uv_hair_protect
            )

        if l_ref is not None:
            outer_rim_dark = (
                v_buf["face_valid"]
                & (uv_x_abs > 0.68)
                & (uv_hair_protect < 0.10)
                & ((l_ref - v_lab_l) > 18.0)
            )
        else:
            outer_rim_dark = np.zeros((atlas_size, atlas_size), dtype=bool)

        is_bad = (v_buf["eff_hair"] > 0.30) | (v_buf["non_skin"] > 0.35) | outer_rim_dark
        clean_v = np.clip(1.0 - 1.25 * v_buf["eff_hair"] - 1.15 * v_buf["non_skin"], 0.02, 1.0).astype(np.float32)
        clean_v = np.where(outer_rim_dark, np.float32(0.02), clean_v).astype(np.float32)

        facing = v_buf["facing"]
        v_res = v_buf["res"]
        if view_key == "front":
            geom_base = (np.power(facing, 2.2) * (1.06 + 2.2 * uv_strict_lock) * v_res).astype(np.float32)
        elif view_key in ("three_quarter", "three_quarter_2"):
            geom_base = (
                np.power(facing, 2.4) * 1.18 * (1.0 - 0.92 * uv_strict_lock)
                * v_buf["far_guard"] * v_res * v_buf["vis"]
            ).astype(np.float32)
        else:
            geom_base = (
                np.power(facing, 2.4) * 1.08 * (1.0 - 0.98 * uv_strict_lock)
                * v_buf["far_guard"] * v_res * v_buf["vis"]
            ).astype(np.float32)

        smooth_geom = _masked_gaussian_blur(geom_base, v_buf["face_valid"], score_blur_sigma)
        v_buf["outer_rim_dark"] = outer_rim_dark
        v_buf["is_bad"] = is_bad
        v_buf["clean"] = clean_v
        v_buf["face_score"] = np.where(v_buf["face_valid"], smooth_geom * clean_v, -1.0).astype(np.float32)
        v_buf["eye_score"] = np.where(v_buf["eye_valid"], geom_base * clean_v, -1.0).astype(np.float32)

    # Step C: Winner-take-all view selection + connected-components island cleanup (<250 px)
    best_face_score = np.full((atlas_size, atlas_size), -1.0, dtype=np.float32)
    face_winner_id = np.full((atlas_size, atlas_size), -1, dtype=np.int8)
    for view_key in active_keys:
        v_buf = view_buffers[view_key]
        win = v_buf["face_valid"] & (v_buf["face_score"] > best_face_score)
        best_face_score[win] = v_buf["face_score"][win]
        face_winner_id[win] = v_buf["idx"]

    active_indices = [view_to_idx[k] for k in active_keys]
    face_winner_id = _cleanup_small_islands(face_winner_id, face_covered, active_indices, min_island_px)

    best_eye_score = np.full((atlas_size, atlas_size), -1.0, dtype=np.float32)
    eye_winner_id = np.full((atlas_size, atlas_size), -1, dtype=np.int8)
    for view_key in active_keys:
        v_buf = view_buffers[view_key]
        win = v_buf["eye_valid"] & (v_buf["eye_score"] > best_eye_score)
        best_eye_score[win] = v_buf["eye_score"][win]
        eye_winner_id[win] = v_buf["idx"]

    winner_id = np.where(face_covered, face_winner_id, np.where(eye_covered, eye_winner_id, -1)).astype(np.int8)
    wta_rgb = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    winner_clean = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    winning_views: list[tuple[str, dict]] = []
    for view_key in active_keys:
        v_buf = view_buffers[view_key]
        m = winner_id == v_buf["idx"]
        if np.any(m):
            wta_rgb[m] = v_buf["rgb"][m]
            winner_clean[m] = v_buf["clean"][m]
            winning_views.append((view_key, v_buf))

    # Part 2: Hair/Shadow rejection evaluated strictly across valid covering views (valid_proj & vis >= 0.15)
    any_covering = np.zeros((atlas_size, atlas_size), dtype=bool)
    has_clean_covering = np.zeros((atlas_size, atlas_size), dtype=bool)
    for view_key in active_keys:
        v_buf = view_buffers[view_key]
        cov_v = v_buf["face_valid"] & (v_buf["vis"] >= 0.15)
        any_covering |= cov_v
        has_clean_covering |= cov_v & (~v_buf["is_bad"])

    rejected_hair = face_covered & any_covering & (~has_clean_covering)
    border_cleaned_px = int(np.sum(rejected_hair))

    fill_rad = max(1, int(round(4.0 * scale)))
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fill_rad + 1, 2 * fill_rad + 1))
    if np.any(rejected_hair):
        dil_rej = (cv2.dilate(rejected_hair.astype(np.uint8), fill_kernel) > 0) & (uv_hair_protect < 0.15)
        fill_mask = (rejected_hair | dil_rej) & face_covered
    else:
        fill_mask = np.zeros((atlas_size, atlas_size), dtype=bool)

    clean_src_mask = face_covered & (face_winner_id >= 0) & (~fill_mask)
    pull_field = _compute_clean_push_pull_field(wta_rgb, clean_src_mask, atlas_size)
    I_base_f32 = _apply_push_pull_fill(
        wta_rgb.astype(np.float32), pull_field, fill_mask, face_covered, uv_hair_protect, atlas_size
    )

    any_covered = face_covered | eye_covered
    valid_covered = (face_covered & (face_winner_id >= 0)) | (eye_covered & (eye_winner_id >= 0)) | fill_mask

    # Step 3: Seam-Only Multi-Band Laplacian Pyramid Blend
    if enable_seam_blend and len(winning_views) > 1:
        if pull_field is not None:
            I_base_pad_f32 = np.where(valid_covered[:, :, None], I_base_f32, pull_field).astype(np.float32)
        else:
            I_base_pad_f32 = I_base_f32
        blended_f32 = _blend_laplacian_pyramids(
            I_base_pad_f32, winning_views, winner_id, any_covered, atlas_size
        )
        blended_f32 = _apply_push_pull_fill(
            blended_f32, pull_field, fill_mask, face_covered, uv_hair_protect, atlas_size
        )
        baked_rgb = np.where(any_covered[:, :, None], np.round(blended_f32), 0.0).clip(0.0, 255.0).astype(np.uint8)
    else:
        baked_rgb = np.where(any_covered[:, :, None], np.round(I_base_f32), 0.0).clip(0.0, 255.0).astype(np.uint8)

    baked_rgb_bled = _dilate_uv_bleed(baked_rgb, valid_covered, radius_px=6)

    # Debug 1: Winner source map (512x512)
    source_full = np.full((atlas_size, atlas_size, 3), UNCOVERED_GREY_RGB, dtype=np.uint8)
    for idx in active_indices:
        m = any_covered & (winner_id == idx)
        if np.any(m):
            source_full[m] = SOURCE_COLORS_RGB[idx]
    debug_source_map = cv2.resize(source_full, (debug_size, debug_size), interpolation=cv2.INTER_NEAREST)

    # Debug 2: Hair rejection & auxiliary recovery map (512x512)
    hair_full = np.full((atlas_size, atlas_size, 3), UNCOVERED_GREY_RGB, dtype=np.uint8)
    if np.any(any_covered):
        luma = (
            0.299 * baked_rgb[:, :, 0].astype(np.float32)
            + 0.587 * baked_rgb[:, :, 1].astype(np.float32)
            + 0.114 * baked_rgb[:, :, 2].astype(np.float32)
        ) * 0.55
        gray_u8 = np.clip(luma, 0.0, 255.0).astype(np.uint8)
        hair_full[any_covered] = np.stack((gray_u8, gray_u8, gray_u8), axis=-1)[any_covered]

    front_rejected = face_covered & f_buf["face_valid"] & f_buf["is_bad"]
    recovered_mask = front_rejected & (winner_id > 0) & (winner_clean > f_buf["clean"] + 0.08)
    unrecovered_mask = front_rejected & (~recovered_mask)
    if np.any(unrecovered_mask):
        hair_full[unrecovered_mask] = HAIR_RED_RGB
    if np.any(recovered_mask):
        hair_full[recovered_mask] = RECOVERED_GREEN_RGB
    debug_hair_mask = cv2.resize(hair_full, (debug_size, debug_size), interpolation=cv2.INTER_NEAREST)

    total_face_px = max(int(np.sum(face_covered & (face_winner_id >= 0))), 1)
    coverage = {}
    for k in VIEW_ORDER:
        idx = view_to_idx[k]
        cnt = int(np.sum(face_covered & (face_winner_id == idx)))
        coverage[k] = round(100.0 * float(cnt) / float(total_face_px), 1)
    hair_replaced_px = int(np.sum(recovered_mask))
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    cm_flag = "ON" if enable_color_match else "OFF"
    sb_flag = "ON" if enable_seam_blend else "OFF"
    print(
        f"[YedpFace3D][UVBaker] Baked {atlas_size}x{atlas_size} in {elapsed_ms:.0f}ms | "
        f"views={len(active_keys)} | "
        f"coverage: front={coverage['front']:.0f}%, 3q={coverage['three_quarter']:.0f}%, "
        f"3q2={coverage['three_quarter_2']:.0f}%, side={coverage['side']:.0f}%, "
        f"side2={coverage['side_2']:.0f}% | "
        f"hair_replaced_px={hair_replaced_px}, border_cleaned_px={border_cleaned_px} | "
        f"color_match={cm_flag}, seam_blend={sb_flag}"
    )

    return {
        "baked_albedo": _encode_png_data_url(baked_rgb_bled),
        "debug_source_map": _encode_png_data_url(debug_source_map),
        "debug_hair_mask": _encode_png_data_url(debug_hair_mask),
        "stats": {
            "atlas_size": atlas_size,
            "elapsed_ms": round(elapsed_ms, 1),
            "views": len(active_keys),
            "coverage": coverage,
            "hair_replaced_px": hair_replaced_px,
            "border_cleaned_px": border_cleaned_px,
            "color_match": enable_color_match,
            "seam_blend": enable_seam_blend,
            "color_shifts_dl": color_shifts_dl,
            "covered_face_px": int(np.sum(face_covered)),
        },
    }


# Verify canonical UV atlas integrity at module startup
validate_uv_atlas(*get_canonical_atlas())
