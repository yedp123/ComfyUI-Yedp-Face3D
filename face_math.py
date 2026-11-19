import numpy as np

# MediaPipe canonical landmark indices used for alignment, scaling, and anatomical anchors
NOSE_BRIDGE_IDX = 168
NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
UPPER_LIP_INNER_IDX = 13
LOWER_LIP_INNER_IDX = 14
LEFT_TEMPLE_IDX = 234
RIGHT_TEMPLE_IDX = 454
CHIN_IDX = 152
FOREHEAD_IDX = 10

# Ordered sagittal midline landmarks (forehead -> nose bridge -> tip -> philtrum -> lips -> chin)
SAGITTAL_PROFILE_INDICES = [
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0,
    11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152,
]

# Bilateral nostril transition sets for smoothing protruding alar wings into the tip/cheek
NOSTRIL_RIM_INDICES = frozenset((
    20, 48, 59, 60, 75, 79, 115, 166, 218, 219, 235, 238, 239, 240,
    250, 278, 289, 290, 305, 309, 344, 392, 438, 439, 455, 458, 459, 460,
))
NOSTRIL_MID_WING_INDICES = frozenset((
    220, 237, 241, 242, 440, 457, 461, 462,
))
NOSTRIL_TIP_COLLAR_INDICES = frozenset((
    44, 45, 125, 141, 274, 275, 354, 370,
))
NOSTRIL_CREASE_INDICES = frozenset((
    49, 51, 64, 97, 98, 99, 102, 129, 131, 134, 198, 209,
    279, 281, 294, 326, 327, 328, 331, 358, 360, 363, 420, 429,
))

TARGET_EYE_MOUTH_SCALE = 0.24
MAX_FACE_MESH_VERTS = 468

# Verified MediaPipe topology sets for Phase 2 anatomical importance groups
FACE_OVAL_INDICES = frozenset((
    10, 109, 67, 103, 54, 21, 162, 127, 234,
    93, 132, 58, 172, 136, 150, 149, 176, 148,
    152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454, 356, 389, 251, 284, 332, 297, 338,
))
JAW_CHIN_INDICES = frozenset((
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    377, 400, 378, 379, 365, 397, 288, 361,
    175, 199, 200, 201, 421, 418, 262, 428, 32, 140, 171, 396, 369,
))
NASAL_BRIDGE_INDICES = frozenset((168, 6, 197, 195, 5, 8, 9))
BROW_INDICES = frozenset((
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
))
EYE_CORNER_INDICES = frozenset((33, 133, 263, 362))
EYE_RIM_FOLD_INDICES = frozenset((
    246, 161, 160, 159, 158, 157, 173, 7, 163, 144, 145, 153, 154, 155,
    27, 28, 29, 30, 56, 190, 247,
    466, 388, 387, 386, 385, 384, 398, 249, 390, 373, 374, 380, 381, 382,
    257, 258, 259, 260, 286, 414, 467,
))
LIP_CONTOUR_INDICES = frozenset((
    0, 11, 12, 13, 37, 38, 39, 40, 41, 42, 62, 72, 73, 74, 76, 77,
    80, 81, 82, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 292,
    302, 303, 304, 306, 307, 310, 311, 312, 407, 408, 409, 415,
    14, 15, 16, 17, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146,
    178, 179, 180, 181, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325,
    375, 402, 403, 404, 405, 61, 78, 291, 308,
))
NOSE_TIP_NOSTRIL_INDICES = (
    frozenset((4, 1, 19, 94, 2, 98, 97, 326, 327))
    | NOSTRIL_RIM_INDICES
    | NOSTRIL_MID_WING_INDICES
    | NOSTRIL_TIP_COLLAR_INDICES
    | NOSTRIL_CREASE_INDICES
)

# Per-landmark sagittal reliability priors for 2D side-profile extraction
SAGITTAL_RELIABILITY_PRIORS = {
    10: 0.25, 151: 0.45,          # Hairline / upper forehead (often hair-occluded)
    9: 0.80, 8: 0.85,             # Glabella / upper brow ridge
    168: 1.0, 6: 1.0, 197: 1.0, 195: 1.0, 5: 1.0, 4: 1.0, 1: 1.0, 19: 1.0, 94: 1.0, 2: 1.0,
    164: 0.90, 0: 0.85,           # Philtrum / upper vermilion border
    11: 0.55, 12: 0.55,           # Upper lip surface
    13: 0.30, 14: 0.30,           # Inner lip gap (sensitive to mouth parting)
    15: 0.55, 16: 0.55,           # Lower lip surface
    17: 0.85, 18: 0.90, 200: 0.90, 199: 0.92, 175: 0.95, 152: 0.95,
}


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _extract_rotation_matrix(matrix_data: list[float] | None) -> np.ndarray:
    """Return an orthonormal 3x3 camera-space rotation matrix from a 4x4 MediaPipe matrix."""
    if not matrix_data or len(matrix_data) != 16:
        return np.eye(3, dtype=np.float32)

    # MediaPipe Tasks Vision outputs a column-major 4x4 array; transpose to row-major.
    mat = np.asarray(matrix_data, dtype=np.float32).reshape(4, 4).T
    rot = mat[:3, :3]
    u, _, vt = np.linalg.svd(rot)
    ortho = u @ vt
    if np.linalg.det(ortho) < 0:
        u[:, -1] *= -1.0
        ortho = u @ vt
    return ortho.astype(np.float32)


def _landmarks_to_metric(landmarks: list[dict], width: int, height: int) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to aspect-corrected right-handed 3D coords."""
    aspect = float(height) / max(float(width), 1.0)
    pts = np.empty((len(landmarks), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = float(lm["x"]) - 0.5
        pts[i, 1] = -(float(lm["y"]) - 0.5) * aspect
        pts[i, 2] = -float(lm["z"])
    return pts


def _landmarks_to_uv(landmarks: list[dict]) -> np.ndarray:
    """Extract WebGL texture UVs (origin bottom-left) from raw image-space landmarks."""
    uvs = np.empty((len(landmarks), 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        uvs[i, 0] = np.clip(float(lm["x"]), 0.0, 1.0)
        uvs[i, 1] = np.clip(1.0 - float(lm["y"]), 0.0, 1.0)
    return uvs


def canonicalize_view(
    landmarks: list[dict],
    matrix_data: list[float] | None,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Center on #168, un-rotate by inverse 3x3 pose, normalize by eye-to-mouth distance,
    and return canonical pose metadata (rot, view_dir, yaw_deg, pitch_deg).
    """
    pts = _landmarks_to_metric(landmarks, width, height)
    uvs = _landmarks_to_uv(landmarks)
    rot = _extract_rotation_matrix(matrix_data)

    # Center on nose bridge (#168) before un-rotating to canonical straight-ahead pose.
    centered = pts - pts[NOSE_BRIDGE_IDX : NOSE_BRIDGE_IDX + 1]
    canonical = centered @ rot  # Equivalent to (rot.T @ centered.T).T

    mid_eye = 0.5 * (canonical[LEFT_EYE_OUTER_IDX] + canonical[RIGHT_EYE_OUTER_IDX])
    mid_mouth = 0.5 * (canonical[UPPER_LIP_INNER_IDX] + canonical[LOWER_LIP_INNER_IDX])
    eye_mouth_dist = float(np.linalg.norm(mid_eye - mid_mouth))
    scale = TARGET_EYE_MOUTH_SCALE / max(eye_mouth_dist, 1e-5)
    canonical *= scale

    view_dir = rot[2, :].astype(np.float32).copy()
    vd_norm = float(np.linalg.norm(view_dir))
    if vd_norm > 1e-6:
        view_dir = (view_dir / vd_norm).astype(np.float32)
    else:
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    yaw_deg = float(np.degrees(np.arctan2(-float(view_dir[0]), float(view_dir[2]))))
    pitch_deg = float(
        np.degrees(np.arctan2(float(view_dir[1]), float(np.hypot(view_dir[0], view_dir[2]))))
    )
    pose_info = {
        "rot": rot,
        "view_dir": view_dir,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
    }
    return canonical.astype(np.float32), uvs.astype(np.float32), pose_info


def _orient_triangles(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Ensure triangle winding produces outward-facing (+Z) canonical normals."""
    oriented = triangles.copy()
    v0 = positions[oriented[:, 0]]
    v1 = positions[oriented[:, 1]]
    v2 = positions[oriented[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    # In canonical space, outward face normals should point mostly +Z (or radially outward from origin)
    radial = (v0 + v1 + v2) / 3.0
    radial[:, 2] += 0.25
    dots = np.sum(face_normals * radial, axis=1)
    flip_mask = dots < 0.0
    oriented[flip_mask, 1], oriented[flip_mask, 2] = oriented[flip_mask, 2], oriented[flip_mask, 1]
    return oriented


def _compute_vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute smooth area-weighted vertex normals."""
    normals = np.zeros_like(positions, dtype=np.float32)
    v0 = positions[triangles[:, 0]]
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-6)
    return normals.astype(np.float32)


def _compute_face_normals(front_pts: np.ndarray, tri_arr: np.ndarray) -> np.ndarray:
    """Compute outward-guaranteed canonical vertex normals on face vertices 0..467 only."""
    n_face = min(len(front_pts), MAX_FACE_MESH_VERTS)
    if n_face == 0 or len(tri_arr) == 0:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n_face, 1))
    face_mask = np.all((tri_arr >= 0) & (tri_arr < n_face), axis=1)
    face_tris = tri_arr[face_mask]
    if len(face_tris) == 0:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n_face, 1))
    oriented = _orient_triangles(front_pts[:n_face], face_tris)
    normals = _compute_vertex_normals(front_pts[:n_face], oriented)
    if n_face > NOSE_TIP_IDX and normals[NOSE_TIP_IDX, 2] < 0.0:
        normals = -normals
    return normals


def _build_mirror_map(front_pts: np.ndarray) -> np.ndarray:
    """Build a mutual-nearest-neighbor left/right mirror map on canonical face vertices 0..467."""
    n_face = min(len(front_pts), MAX_FACE_MESH_VERTS)
    if n_face == 0:
        return np.empty(0, dtype=np.int32)
    face_pts = front_pts[:n_face]
    mirrored = face_pts.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    diff = face_pts[None, :, :] - mirrored[:, None, :]
    d2 = np.sum(diff * diff, axis=2)
    nn = np.argmin(d2, axis=1)
    mirror = np.full(n_face, -1, dtype=np.int32)
    for i in range(n_face):
        j = int(nn[i])
        if int(nn[j]) == i:
            mirror[i] = j
    return mirror


def _compute_boundary_weights(landmarks: list[dict], n_verts: int) -> np.ndarray:
    """Ramp reliability down to 0.1 within 4% of image borders and 0.0 outside [0, 1]."""
    bnd_w = np.ones(n_verts, dtype=np.float32)
    n_lm = min(len(landmarks), n_verts)
    for i in range(n_lm):
        lm = landmarks[i]
        lx = float(lm["x"])
        ly = float(lm["y"])
        if lx < 0.0 or lx > 1.0 or ly < 0.0 or ly > 1.0:
            bnd_w[i] = 0.0
        else:
            edge_dist = min(lx, 1.0 - lx, ly, 1.0 - ly)
            if edge_dist >= 0.04:
                bnd_w[i] = 1.0
            else:
                t = np.clip(edge_dist / 0.04, 0.0, 1.0)
                smooth_t = float(t * t * (3.0 - 2.0 * t))
                bnd_w[i] = 0.1 + 0.9 * smooth_t
    return bnd_w


def _compute_detector_weights(landmarks: list[dict], n_verts: int) -> np.ndarray:
    """Extract optional per-landmark detector confidence, defaulting neutrally to 1.0."""
    det_w = np.ones(n_verts, dtype=np.float32)
    n_lm = min(len(landmarks), n_verts)
    for i in range(n_lm):
        conf = landmarks[i].get("confidence")
        if conf is not None:
            val = float(conf)
            if np.isfinite(val):
                det_w[i] = np.clip(val, 0.0, 1.0)
    return det_w


def _estimate_effective_visibility_pose(
    landmarks: list[dict],
    pose_info: dict,
) -> dict:
    """
    Estimate true camera-facing visibility direction using 2D nose-to-temple asymmetry
    fused with MediaPipe's 4x4 matrix yaw (preserving sign & matrix pitch).
    """
    n_lm = len(landmarks)
    if n_lm <= max(NOSE_TIP_IDX, LEFT_TEMPLE_IDX, RIGHT_TEMPLE_IDX):
        return {
            "vis_view_dir": pose_info["view_dir"],
            "yaw_eff_deg": pose_info["yaw_deg"],
            "yaw_2d_deg": pose_info["yaw_deg"],
            "yaw_mat_deg": pose_info["yaw_deg"],
            "asym": 0.0,
        }

    d_left = float(landmarks[NOSE_TIP_IDX]["x"]) - float(landmarks[LEFT_TEMPLE_IDX]["x"])
    d_right = float(landmarks[RIGHT_TEMPLE_IDX]["x"]) - float(landmarks[NOSE_TIP_IDX]["x"])
    asym = float(np.clip((d_left - d_right) / max(abs(d_left) + abs(d_right), 1e-5), -1.0, 1.0))

    max_yaw_rad = float(np.radians(85.0))
    pivot_rad = float(np.arcsin(0.68))
    if d_left * d_right < 0.0:
        # Nose has crossed past the occluded temple in 2D X; interpolate cleanly from ~43 deg to 85 deg.
        cross_ratio = min(abs(d_left), abs(d_right)) / max(abs(d_left), abs(d_right), 1e-5)
        yaw_2d_rad = float(np.sign(asym) * (pivot_rad + (max_yaw_rad - pivot_rad) * cross_ratio))
    else:
        yaw_2d_rad = float(np.arcsin(np.clip(0.68 * asym, -1.0, 1.0)))
    yaw_2d_rad = float(np.clip(yaw_2d_rad, -max_yaw_rad, max_yaw_rad))

    yaw_mat_rad = float(np.radians(pose_info["yaw_deg"]))
    pitch_rad = float(np.radians(pose_info["pitch_deg"]))

    if abs(asym) > 0.35:
        if np.sign(yaw_mat_rad) == np.sign(yaw_2d_rad) and abs(yaw_mat_rad) > abs(yaw_2d_rad):
            yaw_eff_rad = yaw_mat_rad
        else:
            yaw_eff_rad = yaw_2d_rad
    else:
        yaw_eff_rad = yaw_mat_rad if abs(yaw_mat_rad) >= abs(yaw_2d_rad) else yaw_2d_rad

    cos_p = float(np.cos(pitch_rad))
    vis_view_dir = np.array([
        -float(np.sin(yaw_eff_rad)) * cos_p,
        float(np.sin(pitch_rad)),
        float(np.cos(yaw_eff_rad)) * cos_p,
    ], dtype=np.float32)
    vd_norm = float(np.linalg.norm(vis_view_dir))
    if vd_norm > 1e-6:
        vis_view_dir = (vis_view_dir / vd_norm).astype(np.float32)
    else:
        vis_view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return {
        "vis_view_dir": vis_view_dir,
        "yaw_eff_deg": float(np.degrees(yaw_eff_rad)),
        "yaw_2d_deg": float(np.degrees(yaw_2d_rad)),
        "yaw_mat_deg": float(pose_info["yaw_deg"]),
        "asym": asym,
    }


def _build_anatomical_weights(
    n_verts: int,
    neighbors: list[list[int]],
    structural_importance: float = 1.75,
    secondary_importance: float = 1.0,
    fine_detail_importance: float = 0.65,
    unreliable_importance: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """
    Partition all 478 vertices into 4 verified MediaPipe topology importance groups.
    Iris vertices 468..477 are assigned directly to Fine Facial Details without neighbor ops.
    """
    n_face = min(n_verts, MAX_FACE_MESH_VERTS)
    unreliable = set()
    for seed in (FOREHEAD_IDX, 151, LEFT_TEMPLE_IDX, RIGHT_TEMPLE_IDX):
        if seed < n_face:
            unreliable.add(seed)
            nbrs = neighbors[seed]
            if seed == 151:
                unreliable.update(nb for nb in nbrs if nb != 9)
            else:
                unreliable.update(nbrs)

    iris_set = set(range(MAX_FACE_MESH_VERTS, n_verts))
    fine_detail = (set(EYE_RIM_FOLD_INDICES) | set(LIP_CONTOUR_INDICES) | iris_set) - unreliable

    eye_socket = set(EYE_CORNER_INDICES)
    for v in EYE_RIM_FOLD_INDICES | EYE_CORNER_INDICES:
        if v < n_face:
            eye_socket.update(neighbors[v])
    eye_socket -= (EYE_RIM_FOLD_INDICES | iris_set | NASAL_BRIDGE_INDICES | BROW_INDICES)

    outer_contour = set(FACE_OVAL_INDICES) - unreliable - JAW_CHIN_INDICES
    secondary = (set(BROW_INDICES) | eye_socket | set(NOSE_TIP_NOSTRIL_INDICES) | outer_contour) - unreliable - fine_detail

    anat_mult = np.full(n_verts, float(structural_importance), dtype=np.float32)
    counts = {"structural": 0, "secondary": 0, "fine_detail": 0, "unreliable": 0}
    for i in range(n_verts):
        if i in unreliable:
            anat_mult[i] = float(unreliable_importance)
            counts["unreliable"] += 1
        elif i in fine_detail:
            anat_mult[i] = float(fine_detail_importance)
            counts["fine_detail"] += 1
        elif i in secondary:
            anat_mult[i] = float(secondary_importance)
            counts["secondary"] += 1
        else:
            counts["structural"] += 1

    return anat_mult, counts


def _huber_damp(delta: np.ndarray, huber_delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply smooth excess-sqrt Huber damping to residuals exceeding huber_delta."""
    abs_d = np.abs(delta)
    excess_mask = abs_d > huber_delta
    if not np.any(excess_mask):
        return delta, excess_mask
    out = delta.copy()
    ex_abs = abs_d[excess_mask]
    damped_mag = huber_delta + np.sqrt(huber_delta * (ex_abs - huber_delta) + huber_delta * huber_delta) - huber_delta
    out[excess_mask] = np.sign(delta[excess_mask]) * damped_mag
    return out.astype(np.float32), excess_mask


def _compute_view_reliability(
    landmarks: list[dict],
    pts_v: np.ndarray,
    view_dir_v: np.ndarray,
    front_pts: np.ndarray,
    normals_face: np.ndarray,
    face_height: float,
    is_front: bool,
    yaw_mismatch_rad: float = 0.0,
    is_side: bool = False,
    yaw_eff_deg: float = 0.0,
) -> dict:
    """Compute per-landmark reliability factors (detector, visibility, boundary, consistency)."""
    n_verts = len(front_pts)
    n_face = min(n_verts, MAX_FACE_MESH_VERTS)

    det_w = _compute_detector_weights(landmarks, n_verts)
    bnd_w = _compute_boundary_weights(landmarks, n_verts)

    vis_w = np.zeros(n_verts, dtype=np.float32)
    facing_w = np.zeros(n_verts, dtype=np.float32)
    if n_face > 0:
        facing = normals_face[:n_face] @ view_dir_v
        facing_w[:n_face] = np.clip(facing, 0.0, 1.0)
        vis_w[:n_face] = _smoothstep(0.05, 0.35, facing)

        # Midline nose-bridge self-occlusion culling for side profiles:
        # far eye & far cheek on the occluded side of the nose bridge drop smoothly to 0.0.
        if is_side:
            if abs(yaw_eff_deg) > 1e-4:
                yaw_sign = 1.0 if yaw_eff_deg > 0.0 else -1.0
            else:
                nose_x = float(landmarks[NOSE_TIP_IDX]["x"]) if len(landmarks) > NOSE_TIP_IDX else 0.5
                bridge_x = float(landmarks[NOSE_BRIDGE_IDX]["x"]) if len(landmarks) > NOSE_BRIDGE_IDX else 0.5
                yaw_sign = 1.0 if nose_x >= bridge_x else -1.0
            max_abs_x = max(float(np.max(np.abs(front_pts[:n_face, 0]))), 1e-5)
            hidden_x_norm = (front_pts[:n_face, 0] * yaw_sign) / max_abs_x
            midline_vis = 1.0 - _smoothstep(0.02, 0.14, hidden_x_norm)
            vis_w[:n_face] *= midline_vis
            facing_w[:n_face] *= midline_vis

    if n_verts >= 473 and n_face > 159:
        vis_w[468:473] = 0.5 * (vis_w[159] + vis_w[145])
        facing_w[468:473] = 0.5 * (facing_w[159] + facing_w[145])
    if n_verts >= 478 and n_face > 386:
        vis_w[473:478] = 0.5 * (vis_w[386] + vis_w[374])
        facing_w[473:478] = 0.5 * (facing_w[386] + facing_w[374])

    if is_front:
        cons_w = np.ones(n_verts, dtype=np.float32)
    else:
        inv_h = 1.0 / max(face_height, 1e-4)
        err_y = np.abs(pts_v[:, 1] - front_pts[:, 1]) * inv_h
        err_x = np.abs(pts_v[:, 0] - front_pts[:, 0]) * inv_h
        # Keep vertical alignment tight (catches open-mouth/pose mismatch) while tolerating
        # matrix-compressed horizontal foreshortening proportional to yaw mismatch.
        x_slack = 0.14 * float(np.sin(min(abs(yaw_mismatch_rad), 1.48)))
        cons_y = 1.0 - _smoothstep(0.05, 0.14, err_y)
        cons_x = 1.0 - _smoothstep(0.06 + x_slack, 0.16 + x_slack * 1.5, err_x)
        cons_w = (cons_y * cons_x).astype(np.float32)

    w_v = np.clip(det_w * vis_w * bnd_w * cons_w, 0.0, 1.0).astype(np.float32)
    return {
        "w": w_v,
        "det_w": det_w,
        "vis_w": vis_w,
        "facing_w": facing_w,
        "bnd_w": bnd_w,
        "cons_w": cons_w,
    }


def _build_neighbors(n_verts: int, triangles: np.ndarray) -> list[list[int]]:
    """Build 1-ring vertex adjacency lists from mesh triangles."""
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for a, b, c in triangles:
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].update((bi, ci))
        adj[bi].update((ai, ci))
        adj[ci].update((ai, bi))
    return [sorted(s) for s in adj]


def _laplacian_smooth_z(
    positions: np.ndarray,
    neighbors: list[list[int]],
    iterations: int = 1,
) -> np.ndarray:
    """Apply gentle Laplacian smoothing to Z only to eliminate multi-view blend seams."""
    out = positions.copy()
    n_face = min(len(out), MAX_FACE_MESH_VERTS)
    for _ in range(iterations):
        z_new = out[:, 2].copy()
        for i in range(n_face):
            nbrs = neighbors[i]
            if nbrs:
                mean_z = float(np.mean(out[nbrs, 2]))
                z_new[i] = 0.85 * out[i, 2] + 0.15 * mean_z
        out[:, 2] = z_new
    return out


def _smooth_nostril_transition(
    positions: np.ndarray,
    neighbors: list[list[int]],
    iterations: int = 4,
) -> np.ndarray:
    """
    Fair the protruding alar wings and nostril rims smoothly into the nose tip
    and cheek crease while keeping the midline tip silhouette (#4, #1, #19, #94, #2) pinned.
    """
    out = positions.copy()
    weights: dict[int, tuple[float, float]] = {}
    for idx in NOSTRIL_CREASE_INDICES:
        weights[idx] = (0.28, 0.0)
    for idx in NOSTRIL_TIP_COLLAR_INDICES:
        weights[idx] = (0.20, 0.0)
    for idx in NOSTRIL_MID_WING_INDICES:
        weights[idx] = (0.34, 0.10)
    for idx in NOSTRIL_RIM_INDICES:
        weights[idx] = (0.58, 0.22)

    n_face = min(len(out), MAX_FACE_MESH_VERTS)
    active = [(idx, wz, wxy) for idx, (wz, wxy) in weights.items() if idx < n_face and neighbors[idx]]
    if not active:
        return out

    for _ in range(iterations):
        nxt = out.copy()
        for idx, wz, wxy in active:
            mean_pt = np.mean(out[neighbors[idx]], axis=0)
            if wxy > 0.0:
                nxt[idx, 0] = (1.0 - wxy) * out[idx, 0] + wxy * mean_pt[0]
                nxt[idx, 1] = (1.0 - wxy) * out[idx, 1] + wxy * mean_pt[1]
            nxt[idx, 2] = (1.0 - wz) * out[idx, 2] + wz * mean_pt[2]
        out = nxt
    return out


def _apply_side_2d_profile(
    merged: np.ndarray,
    neighbors: list[list[int]],
    side_landmarks: list[dict],
    side_width: int,
    side_height: int,
    enable_reliability_weighting: bool = True,
    side_rel: dict | None = None,
    anat_mult: np.ndarray | None = None,
    secondary_importance: float = 1.0,
    huber_delta: float | None = None,
    stats_out: dict | None = None,
) -> np.ndarray:
    """
    Extract exact 2D (Z, Y) sagittal silhouette from a side-profile photo by exact
    landmark ID (never shifted by Y-interpolation) and diffuse smoothly across the
    central nose/lips/chin ridge via Graph Laplacian propagation.
    """
    aspect = float(side_height) / max(float(side_width), 1.0)
    nose_x = float(side_landmarks[NOSE_TIP_IDX]["x"])
    bridge_x = float(side_landmarks[NOSE_BRIDGE_IDX]["x"])
    bridge_y = float(side_landmarks[NOSE_BRIDGE_IDX]["y"])

    # Determine facing direction (+1 if nose points toward +X/right in photo, -1 if left)
    facing_sign = 1.0 if nose_x >= bridge_x else -1.0

    front_span_y = abs(merged[NOSE_BRIDGE_IDX, 1] - merged[CHIN_IDX, 1])
    if front_span_y < 1e-4:
        front_mouth_y = 0.5 * (merged[UPPER_LIP_INNER_IDX, 1] + merged[LOWER_LIP_INNER_IDX, 1])
        front_span_y = abs(merged[NOSE_BRIDGE_IDX, 1] - front_mouth_y)

    side_chin_y = -(float(side_landmarks[CHIN_IDX]["y"]) - bridge_y) * aspect
    side_chin_z = facing_sign * (float(side_landmarks[CHIN_IDX]["x"]) - bridge_x)
    side_span = float(np.hypot(side_chin_z, side_chin_y))
    if side_span < 1e-4:
        return merged

    scale = front_span_y / side_span

    # Pitch-align side 2D (Z, Y) profile vector #168 -> #152 with canonical front #168 -> #152
    front_dy = merged[CHIN_IDX, 1] - merged[NOSE_BRIDGE_IDX, 1]
    front_dz = merged[CHIN_IDX, 2] - merged[NOSE_BRIDGE_IDX, 2]
    d_pitch = float(np.clip(
        np.arctan2(front_dz, -front_dy) - np.arctan2(side_chin_z, -side_chin_y),
        -0.44,
        0.44,
    ))
    cos_p = float(np.cos(d_pitch))
    sin_p = float(np.sin(d_pitch))

    n_verts = len(merged)
    n_face = min(n_verts, MAX_FACE_MESH_VERTS)
    delta_z = np.zeros(n_verts, dtype=np.float32)
    sag_alpha = np.zeros(n_verts, dtype=np.float32)
    det_w = side_rel["det_w"] if side_rel is not None else None
    bnd_w = side_rel["bnd_w"] if side_rel is not None else None

    for idx in SAGITTAL_PROFILE_INDICES:
        if idx >= len(side_landmarks) or idx >= n_face:
            continue
        raw_z = facing_sign * (float(side_landmarks[idx]["x"]) - bridge_x) * scale
        raw_y = -(float(side_landmarks[idx]["y"]) - bridge_y) * aspect * scale
        rot_z = raw_z * cos_p - raw_y * sin_p
        rot_y = raw_z * sin_p + raw_y * cos_p
        delta_z[idx] = (rot_z + merged[NOSE_BRIDGE_IDX, 2]) - merged[idx, 2]

        if enable_reliability_weighting:
            prior = SAGITTAL_RELIABILITY_PRIORS.get(idx, 0.75)
            d_factor = float(det_w[idx]) if det_w is not None else 1.0
            b_factor = float(bnd_w[idx]) if bnd_w is not None else 1.0
            err_y = abs((rot_y + merged[NOSE_BRIDGE_IDX, 1]) - float(merged[idx, 1])) / max(front_span_y, 1e-4)
            y_cons = float(1.0 - _smoothstep(0.04, 0.12, np.asarray(err_y, dtype=np.float32)))
            anat_factor = 1.0
            if anat_mult is not None:
                anat_factor = min(1.0, float(anat_mult[idx]) / max(float(secondary_importance), 1e-4))
            sag_alpha[idx] = float(np.clip(prior * d_factor * b_factor * y_cons * anat_factor, 0.0, 1.0))
        else:
            sag_alpha[idx] = 1.0

    if huber_delta is not None and huber_delta > 0.0:
        damped_dz, damped_mask = _huber_damp(delta_z, float(huber_delta))
        if stats_out is not None:
            stats_out["huber_damped"] = int(stats_out.get("huber_damped", 0)) + int(
                np.sum(damped_mask & (sag_alpha > 0.0))
            )
        delta_z = damped_dz

    diffused = delta_z.copy()
    for _ in range(5):
        nxt = diffused.copy()
        for i in range(n_face):
            nbrs = neighbors[i]
            if not nbrs:
                continue
            alpha = float(sag_alpha[i])
            if not enable_reliability_weighting and alpha > 0.0:
                continue
            nbr_mean = float(np.mean(diffused[nbrs]))
            nxt[i] = alpha * float(delta_z[i]) + (1.0 - alpha) * nbr_mean if alpha > 0.0 else nbr_mean
        diffused = nxt

    out = merged.copy()
    half_width = max(float(np.max(np.abs(merged[:n_face, 0]))), 1e-4)
    abs_x_norm = np.abs(merged[:n_face, 0]) / half_width
    ridge_weight = 1.0 - _smoothstep(0.10, 0.42, abs_x_norm)
    out[:n_face, 2] += diffused[:n_face] * ridge_weight * 0.85
    return out


def _build_inverse_edge_weights(
    front_pts: np.ndarray,
    neighbors: list[list[int]],
    n_face: int,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Precompute normalized inverse-edge-length neighbor weights for face vertices 0..467."""
    edge_weights: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_face
    for i in range(n_face):
        nbrs = neighbors[i]
        if not nbrs:
            continue
        nbr_idx = np.asarray(nbrs, dtype=np.int32)
        diffs = front_pts[nbr_idx] - front_pts[i]
        dists = np.maximum(np.linalg.norm(diffs, axis=1), 1e-4)
        inv_w = (1.0 / dists).astype(np.float32)
        w_sum = float(np.sum(inv_w))
        if w_sum > 1e-8:
            edge_weights[i] = (nbr_idx, inv_w / w_sum)
    return edge_weights


def _regularize_displacement(
    merged: np.ndarray,
    front_pts: np.ndarray,
    neighbors: list[list[int]],
    obs_conf: np.ndarray,
    face_height: float,
    mirror_map: np.ndarray | None = None,
    base_geometry_anchoring: float = 0.25,
    surface_deformation_smoothness: float = 0.45,
    max_local_displacement: float = 0.18,
    symmetry_strength: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Phase 3 displacement-field regularization (preserves base anatomy on 0..467; skips iris 468..477):
    1. Base-shape anchoring proportional to missing observation confidence.
    2. Inverse-edge-length weighted Laplacian smoothing of Z displacement only.
    3. Scale-relative tanh soft-clamping relative to face height.
    4. Optional bilateral symmetry prior modulated by inverse local confidence.
    """
    n_verts = len(merged)
    n_face = min(n_verts, MAX_FACE_MESH_VERTS)
    delta_z = (merged[:, 2] - front_pts[:, 2]).astype(np.float32)

    # 1. Base-Shape Anchoring
    b_anchor = float(np.clip(base_geometry_anchoring, 0.0, 1.0))
    if b_anchor > 0.0:
        anchor_pull = b_anchor * (1.0 - 0.75 * np.clip(obs_conf, 0.0, 1.0))
        delta_z *= (1.0 - anchor_pull)

    # 2. Displacement-only inverse-edge-length weighted Graph Laplacian (0..467 only)
    s_smooth = float(np.clip(surface_deformation_smoothness, 0.0, 1.0))
    if s_smooth > 0.0 and n_face > 0:
        edge_weights = _build_inverse_edge_weights(front_pts, neighbors, n_face)
        nose_ridge = NASAL_BRIDGE_INDICES | frozenset((4, 1, 19, 94, 2))
        fine_edges = EYE_RIM_FOLD_INDICES | LIP_CONTOUR_INDICES
        reg_smooth_w = np.ones(n_face, dtype=np.float32)
        for idx in range(n_face):
            if idx in nose_ridge:
                reg_smooth_w[idx] = 0.30
            elif idx in fine_edges:
                reg_smooth_w[idx] = 0.45

        cur = delta_z[:n_face].copy()
        for _ in range(3):
            nxt = cur.copy()
            for i in range(n_face):
                ew = edge_weights[i]
                if ew is None:
                    continue
                nbr_idx, weights = ew
                alpha = 0.5 * s_smooth * float(reg_smooth_w[i])
                nbr_avg = float(np.dot(cur[nbr_idx], weights))
                nxt[i] = (1.0 - alpha) * cur[i] + alpha * nbr_avg
            cur = nxt
        delta_z[:n_face] = cur

    # 3. Scale-Relative Soft Displacement Clamp
    max_disp_ratio = float(np.clip(max_local_displacement, 0.05, 0.50))
    limit = max(max_disp_ratio * max(face_height, 1e-4), 1e-5)
    raw_max_ratio = float(np.max(np.abs(delta_z[:n_face])) / max(face_height, 1e-4)) if n_face > 0 else 0.0
    soft_clamped = int(np.sum(np.abs(delta_z[:n_face]) > 0.75 * limit)) if n_face > 0 else 0
    delta_z = (limit * np.tanh(delta_z / limit)).astype(np.float32)

    # 4. Optional Bilateral Symmetry Prior (0..467 only)
    sym = float(np.clip(symmetry_strength, 0.0, 1.0))
    if sym > 0.0 and n_face > 0:
        if mirror_map is None:
            mirror_map = _build_mirror_map(front_pts)
        nxt_sym = delta_z[:n_face].copy()
        for i in range(n_face):
            m = int(mirror_map[i])
            if m >= 0 and m != i:
                w_sym = 0.5 * sym * (1.0 - 0.60 * float(np.clip(obs_conf[i], 0.0, 1.0)))
                nxt_sym[i] = (1.0 - w_sym) * delta_z[i] + w_sym * delta_z[m]
        delta_z[:n_face] = nxt_sym

    out = merged.copy()
    out[:, 2] = front_pts[:, 2] + delta_z
    reg_stats = {
        "max_delta_ratio": round(raw_max_ratio, 4),
        "soft_clamped": soft_clamped,
    }
    return out, reg_stats


def reconstruct_face_mesh(
    front: dict,
    triangles: list[int],
    three_quarter: dict | None = None,
    side: dict | None = None,
    depth_weight: float = 1.0,
    enable_reliability_weighting: bool = True,
    mirror_transfer_weight: float = 0.85,
    enable_anatomical_weighting: bool = True,
    structural_importance: float = 1.75,
    secondary_importance: float = 1.0,
    fine_detail_importance: float = 0.65,
    unreliable_importance: float = 0.10,
    huber_delta_ratio: float = 0.08,
    enable_regularization: bool = True,
    base_geometry_anchoring: float = 0.25,
    surface_deformation_smoothness: float = 0.45,
    max_local_displacement: float = 0.18,
    symmetry_strength: float = 0.0,
) -> dict:
    """
    Reconstruct a canonical 468/478-vertex textured face mesh from 1-3 MediaPipe views.

    Each view dict expects: {"landmarks": [...], "matrix": [...]|None, "width": int, "height": int}.
    """
    front_pts, uv_front, front_pose = canonicalize_view(
        front["landmarks"],
        front.get("matrix"),
        int(front.get("width", 512)),
        int(front.get("height", 512)),
    )
    vis_pose_front = _estimate_effective_visibility_pose(front["landmarks"], front_pose)

    # Restrict triangulation & normal operations strictly to face mesh vertices (0..467)
    n_verts = len(front_pts)
    n_face = min(n_verts, MAX_FACE_MESH_VERTS)
    tri_arr = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    valid_tri_mask = np.all((tri_arr >= 0) & (tri_arr < n_face), axis=1)
    tri_arr = tri_arr[valid_tri_mask]
    neighbors = _build_neighbors(n_verts, tri_arr)
    normals_face = _compute_face_normals(front_pts, tri_arr)

    face_height = max(float(np.linalg.norm(front_pts[FOREHEAD_IDX] - front_pts[CHIN_IDX])), 1e-4)
    rel_front = _compute_view_reliability(
        front["landmarks"],
        front_pts,
        vis_pose_front["vis_view_dir"],
        front_pts,
        normals_face,
        face_height,
        is_front=True,
    )

    # Front view drives X/Y and texture; 3/4 and Side views refine 3D Z depth only.
    merged = front_pts.copy()
    half_width = max(float(np.max(np.abs(front_pts[:, 0]))), 1e-4)
    abs_x_norm = np.abs(front_pts[:, 0]) / half_width

    has_3q = three_quarter is not None and bool(three_quarter.get("landmarks"))
    pts_3q = None
    pose_3q = None
    vis_pose_3q = None
    rel_3q = None
    if has_3q:
        pts_3q, _, pose_3q = canonicalize_view(
            three_quarter["landmarks"],
            three_quarter.get("matrix"),
            int(three_quarter.get("width", 512)),
            int(three_quarter.get("height", 512)),
        )
        vis_pose_3q = _estimate_effective_visibility_pose(three_quarter["landmarks"], pose_3q)
        rel_3q = _compute_view_reliability(
            three_quarter["landmarks"],
            pts_3q,
            vis_pose_3q["vis_view_dir"],
            front_pts,
            normals_face,
            face_height,
            is_front=False,
            yaw_mismatch_rad=np.radians(vis_pose_3q["yaw_eff_deg"] - vis_pose_3q["yaw_mat_deg"]),
            is_side=False,
            yaw_eff_deg=vis_pose_3q["yaw_eff_deg"],
        )

    has_side = side is not None and bool(side.get("landmarks"))
    pts_side = None
    pose_side = None
    vis_pose_side = None
    rel_side = None
    if has_side:
        pts_side, _, pose_side = canonicalize_view(
            side["landmarks"],
            side.get("matrix"),
            int(side.get("width", 512)),
            int(side.get("height", 512)),
        )
        vis_pose_side = _estimate_effective_visibility_pose(side["landmarks"], pose_side)
        rel_side = _compute_view_reliability(
            side["landmarks"],
            pts_side,
            vis_pose_side["vis_view_dir"],
            front_pts,
            normals_face,
            face_height,
            is_front=False,
            yaw_mismatch_rad=np.radians(vis_pose_side["yaw_eff_deg"] - vis_pose_side["yaw_mat_deg"]),
            is_side=True,
            yaw_eff_deg=vis_pose_side["yaw_eff_deg"],
        )

    occluded_3q = int(np.sum(rel_3q["vis_w"][:n_face] < 0.15)) if rel_3q is not None else 0
    occluded_side = int(np.sum(rel_side["vis_w"][:n_face] < 0.15)) if rel_side is not None else 0
    mirrored_transfers = 0
    huber_stats = {"huber_damped": 0}
    mirror_map = _build_mirror_map(front_pts) if (has_3q or has_side) else None

    anat_mult, anat_group_counts = _build_anatomical_weights(
        n_verts,
        neighbors,
        structural_importance=structural_importance,
        secondary_importance=secondary_importance,
        fine_detail_importance=fine_detail_importance,
        unreliable_importance=unreliable_importance,
    )
    use_anat = bool(enable_reliability_weighting and enable_anatomical_weighting)
    huber_delta = max(float(huber_delta_ratio) * face_height, 1e-6) if use_anat else None

    if enable_reliability_weighting:
        mirror_w = float(np.clip(mirror_transfer_weight, 0.0, 1.0))

        w_front_eff = np.maximum(rel_front["w"] * (0.35 + 0.65 * rel_front["facing_w"]), 1e-4)
        z_num = w_front_eff * front_pts[:, 2]
        w_den = w_front_eff.copy()

        for pts_v, rel_v in ((pts_3q, rel_3q), (pts_side, rel_side)):
            if pts_v is None or rel_v is None:
                continue
            vis_v = rel_v["vis_w"]
            w_v = rel_v["w"]
            fusion_v = w_v * (0.35 + 0.65 * rel_v["facing_w"])
            z_v = pts_v[:, 2].copy()
            w_eff = fusion_v.copy()

            if mirror_map is not None and mirror_w > 0.0:
                for i in range(n_face):
                    if vis_v[i] < 0.15:
                        m = int(mirror_map[i])
                        if m >= 0 and m != i and vis_v[m] > 0.5:
                            partner_delta_z = float(pts_v[m, 2] - front_pts[m, 2])
                            z_v[i] = float(front_pts[i, 2]) + mirror_w * partner_delta_z
                            w_eff[i] = float(fusion_v[m]) * mirror_w
                            if w_eff[i] > 0.0:
                                mirrored_transfers += 1

            if use_anat:
                raw_delta = z_v - front_pts[:, 2]
                damped_delta, damped_mask = _huber_damp(raw_delta, huber_delta)
                huber_stats["huber_damped"] += int(np.sum(damped_mask & (w_eff > 0.0)))
                z_v = front_pts[:, 2] + damped_delta
                # Scale secondary-view override authority by anatomical group importance
                # (structural 1.75x strong vs. fine details 0.65x / unreliable 0.10x front-anchored).
                w_eff = w_eff * anat_mult

            z_num += w_eff * z_v
            w_den += w_eff

        merged[:, 2] = z_num / np.maximum(w_den, 1e-6)

        if has_side:
            merged = _apply_side_2d_profile(
                merged,
                neighbors,
                side["landmarks"],
                int(side.get("width", 512)),
                int(side.get("height", 512)),
                enable_reliability_weighting=True,
                side_rel=rel_side,
                anat_mult=anat_mult if use_anat else None,
                secondary_importance=secondary_importance,
                huber_delta=huber_delta,
                stats_out=huber_stats,
            )

        front_dbg = rel_front["w"].astype(np.float32)
        tq_dbg = rel_3q["w"].astype(np.float32) if rel_3q is not None else None
        side_dbg = rel_side["w"].astype(np.float32) if rel_side is not None else None

        combined_weights = front_dbg.copy()
        if tq_dbg is not None:
            combined_weights = np.maximum(combined_weights, tq_dbg)
        if side_dbg is not None:
            combined_weights = np.maximum(combined_weights, side_dbg)
        combined_weights = np.clip(combined_weights, 0.0, 1.0).astype(np.float32)
    else:
        if has_3q:
            w_3q = _smoothstep(0.16, 0.50, abs_x_norm) * (1.0 - 0.5 * _smoothstep(0.75, 1.0, abs_x_norm))
            merged[:, 2] = merged[:, 2] * (1.0 - 0.65 * w_3q) + pts_3q[:, 2] * (0.65 * w_3q)

        if has_side:
            outer_jaw = _smoothstep(0.55, 0.90, abs_x_norm)
            w_jaw = np.clip(outer_jaw * 0.70, 0.0, 1.0)
            merged[:, 2] = merged[:, 2] * (1.0 - w_jaw) + pts_side[:, 2] * w_jaw

            merged = _apply_side_2d_profile(
                merged,
                neighbors,
                side["landmarks"],
                int(side.get("width", 512)),
                int(side.get("height", 512)),
                enable_reliability_weighting=False,
            )

        combined_weights = np.ones(n_verts, dtype=np.float32)
        front_dbg = combined_weights
        tq_dbg = np.ones(n_verts, dtype=np.float32) if has_3q else None
        side_dbg = np.ones(n_verts, dtype=np.float32) if has_side else None

    reg_stats = {"max_delta_ratio": 0.0, "soft_clamped": 0}
    if enable_regularization and (has_3q or has_side):
        obs_conf = rel_front["w"].copy()
        if rel_3q is not None:
            obs_conf = np.maximum(obs_conf, rel_3q["w"])
        if rel_side is not None:
            obs_conf = np.maximum(obs_conf, rel_side["w"])
        if use_anat:
            anat_conf = np.minimum(1.0, anat_mult / max(float(secondary_importance), 1e-4))
            obs_conf = np.clip(obs_conf * anat_conf, 0.0, 1.0)
        merged, reg_stats = _regularize_displacement(
            merged=merged,
            front_pts=front_pts,
            neighbors=neighbors,
            obs_conf=obs_conf,
            face_height=face_height,
            mirror_map=mirror_map,
            base_geometry_anchoring=base_geometry_anchoring,
            surface_deformation_smoothness=surface_deformation_smoothness,
            max_local_displacement=max_local_displacement,
            symmetry_strength=symmetry_strength,
        )
    elif has_3q or has_side:
        merged = _laplacian_smooth_z(merged, neighbors, iterations=1)

    merged = _smooth_nostril_transition(merged, neighbors, iterations=4)
    merged -= merged[NOSE_BRIDGE_IDX : NOSE_BRIDGE_IDX + 1]

    # Unify nasal tip apex (#4, #1) so the center tip never indents into a split-cartilage cleft
    if len(merged) > 275:
        wing_max_z = float(np.max(merged[[44, 45, 274, 275], 2]))
        merged[4, 2] = max(float(merged[4, 2]), wing_max_z + 0.004)
        merged[1, 2] = max(float(merged[1, 2]), 0.5 * (float(merged[4, 2]) + float(merged[19, 2])))

    # Apply depth_weight to Z relative to nose bridge anchor (#168, which is at 0,0,0)
    merged[:, 2] *= float(np.clip(depth_weight, 0.0, 2.5))

    tri_arr = _orient_triangles(merged, tri_arr)

    # Extract anatomical guide anchors directly from canonical vertices
    left_temple = merged[LEFT_TEMPLE_IDX].tolist()
    right_temple = merged[RIGHT_TEMPLE_IDX].tolist()
    chin = merged[CHIN_IDX].tolist()
    forehead = merged[FOREHEAD_IDX].tolist()

    dot_234_3q = (
        float(np.dot(normals_face[LEFT_TEMPLE_IDX], vis_pose_3q["vis_view_dir"]))
        if vis_pose_3q is not None and n_face > LEFT_TEMPLE_IDX
        else None
    )
    dot_454_3q = (
        float(np.dot(normals_face[RIGHT_TEMPLE_IDX], vis_pose_3q["vis_view_dir"]))
        if vis_pose_3q is not None and n_face > RIGHT_TEMPLE_IDX
        else None
    )
    dot_234_side = (
        float(np.dot(normals_face[LEFT_TEMPLE_IDX], vis_pose_side["vis_view_dir"]))
        if vis_pose_side is not None and n_face > LEFT_TEMPLE_IDX
        else None
    )
    dot_454_side = (
        float(np.dot(normals_face[RIGHT_TEMPLE_IDX], vis_pose_side["vis_view_dir"]))
        if vis_pose_side is not None and n_face > RIGHT_TEMPLE_IDX
        else None
    )

    huber_damped = int(huber_stats["huber_damped"])
    max_delta_ratio = float(reg_stats["max_delta_ratio"])
    soft_clamped = int(reg_stats["soft_clamped"])
    print(
        f"[YedpFace3D][Phase3] side_occluded={occluded_side}, mirrored_transfers={mirrored_transfers} | "
        f"huber_damped={huber_damped} | reg_on={bool(enable_regularization)}, "
        f"max_delta_ratio={max_delta_ratio:.4f}, soft_clamped={soft_clamped}"
    )

    # Count-normalized anatomical group summary (prevents point-count bias in global statistics)
    anat_group_means = {
        "structural": round(float(structural_importance), 3),
        "secondary": round(float(secondary_importance), 3),
        "fine_detail": round(float(fine_detail_importance), 3),
        "unreliable": round(float(unreliable_importance), 3),
    }

    weight_debug_stats = {
        "enabled": bool(enable_reliability_weighting),
        "anatomical_enabled": bool(use_anat),
        "regularization_enabled": bool(enable_regularization),
        "regularization": {
            "base_geometry_anchoring": round(float(base_geometry_anchoring), 3),
            "surface_deformation_smoothness": round(float(surface_deformation_smoothness), 3),
            "max_local_displacement": round(float(max_local_displacement), 3),
            "symmetry_strength": round(float(symmetry_strength), 3),
            "max_delta_ratio": max_delta_ratio,
            "soft_clamped": soft_clamped,
        },
        "view_yaws_deg": {
            "front": round(vis_pose_front["yaw_eff_deg"], 2),
            "three_quarter": round(vis_pose_3q["yaw_eff_deg"], 2) if vis_pose_3q is not None else None,
            "side": round(vis_pose_side["yaw_eff_deg"], 2) if vis_pose_side is not None else None,
        },
        "matrix_yaws_deg": {
            "front": round(vis_pose_front["yaw_mat_deg"], 2),
            "three_quarter": round(vis_pose_3q["yaw_mat_deg"], 2) if vis_pose_3q is not None else None,
            "side": round(vis_pose_side["yaw_mat_deg"], 2) if vis_pose_side is not None else None,
        },
        "view_pitches_deg": {
            "front": round(front_pose["pitch_deg"], 2),
            "three_quarter": round(pose_3q["pitch_deg"], 2) if pose_3q is not None else None,
            "side": round(pose_side["pitch_deg"], 2) if pose_side is not None else None,
        },
        "temple_dots_3q": (
            {"left_234": round(dot_234_3q, 4), "right_454": round(dot_454_3q, 4)}
            if dot_234_3q is not None and dot_454_3q is not None
            else None
        ),
        "temple_dots_side": (
            {"left_234": round(dot_234_side, 4), "right_454": round(dot_454_side, 4)}
            if dot_234_side is not None and dot_454_side is not None
            else None
        ),
        "occluded_3q": occluded_3q,
        "occluded_side": occluded_side,
        "mirrored_transfers": mirrored_transfers,
        "huber_damped": huber_damped,
        "anatomical_group_counts": anat_group_counts,
        "anatomical_importance": anat_group_means,
        "mean_weight": round(float(np.mean(combined_weights)), 4),
        "min_weight": round(float(np.min(combined_weights)), 4),
        "max_weight": round(float(np.max(combined_weights)), 4),
        "bucket_counts": {
            "green": int(np.sum(combined_weights > 0.70)),
            "yellow": int(np.sum((combined_weights >= 0.30) & (combined_weights <= 0.70))),
            "red": int(np.sum((combined_weights >= 0.05) & (combined_weights < 0.30))),
            "grey": int(np.sum(combined_weights < 0.05)),
        },
    }

    return {
        "positions": merged.astype(np.float32).ravel().tolist(),
        "uv_front": uv_front.astype(np.float32).ravel().tolist(),
        "indices": tri_arr.astype(np.int32).ravel().tolist(),
        "has_three_quarter": has_3q,
        "has_side": has_side,
        "anchors": {
            "left_temple": left_temple,
            "right_temple": right_temple,
            "chin": chin,
            "forehead": forehead,
        },
        "landmark_weights": combined_weights.tolist(),
        "landmark_weights_by_view": {
            "combined": combined_weights.tolist(),
            "front": front_dbg.tolist(),
            "three_quarter": tq_dbg.tolist() if tq_dbg is not None else None,
            "side": side_dbg.tolist() if side_dbg is not None else None,
        },
        "weight_debug_stats": weight_debug_stats,
    }
