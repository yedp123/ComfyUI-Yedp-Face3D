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
) -> tuple[np.ndarray, np.ndarray]:
    """Center on #168, un-rotate by inverse 3x3 pose, and normalize by eye-to-mouth distance."""
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

    return canonical.astype(np.float32), uvs.astype(np.float32)


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
    for _ in range(iterations):
        z_new = out[:, 2].copy()
        for i, nbrs in enumerate(neighbors):
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

    n_verts = len(out)
    active = [(idx, wz, wxy) for idx, (wz, wxy) in weights.items() if idx < n_verts and neighbors[idx]]
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
    delta_z = np.zeros(n_verts, dtype=np.float32)
    pinned = np.zeros(n_verts, dtype=bool)
    for idx in SAGITTAL_PROFILE_INDICES:
        if idx >= len(side_landmarks) or idx >= n_verts:
            continue
        raw_z = facing_sign * (float(side_landmarks[idx]["x"]) - bridge_x) * scale
        raw_y = -(float(side_landmarks[idx]["y"]) - bridge_y) * aspect * scale
        rot_z = raw_z * cos_p - raw_y * sin_p
        delta_z[idx] = (rot_z + merged[NOSE_BRIDGE_IDX, 2]) - merged[idx, 2]
        pinned[idx] = True

    diffused = delta_z.copy()
    for _ in range(5):
        nxt = diffused.copy()
        for i, nbrs in enumerate(neighbors):
            if pinned[i] or not nbrs:
                continue
            nxt[i] = float(np.mean(diffused[nbrs]))
        diffused = nxt

    out = merged.copy()
    half_width = max(float(np.max(np.abs(merged[:, 0]))), 1e-4)
    abs_x_norm = np.abs(merged[:, 0]) / half_width
    ridge_weight = 1.0 - _smoothstep(0.10, 0.42, abs_x_norm)
    out[:, 2] += diffused * ridge_weight * 0.85
    return out


def reconstruct_face_mesh(
    front: dict,
    triangles: list[int],
    three_quarter: dict | None = None,
    side: dict | None = None,
    depth_weight: float = 1.0,
) -> dict:
    """
    Reconstruct a canonical 468/478-vertex textured face mesh from 1-3 MediaPipe views.

    Each view dict expects: {"landmarks": [...], "matrix": [...]|None, "width": int, "height": int}.
    """
    front_pts, uv_front = canonicalize_view(
        front["landmarks"],
        front.get("matrix"),
        int(front.get("width", 512)),
        int(front.get("height", 512)),
    )

    # Restrict triangulation indices to valid face mesh vertices (0..467)
    n_verts = len(front_pts)
    tri_arr = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    valid_tri_mask = np.all((tri_arr >= 0) & (tri_arr < n_verts), axis=1)
    tri_arr = tri_arr[valid_tri_mask]
    neighbors = _build_neighbors(n_verts, tri_arr)

    # Front view drives X/Y and texture; 3/4 and Side views refine 3D Z depth only.
    merged = front_pts.copy()
    half_width = max(float(np.max(np.abs(front_pts[:, 0]))), 1e-4)
    abs_x_norm = np.abs(front_pts[:, 0]) / half_width

    has_3q = three_quarter is not None and bool(three_quarter.get("landmarks"))
    if has_3q:
        pts_3q, _ = canonicalize_view(
            three_quarter["landmarks"],
            three_quarter.get("matrix"),
            int(three_quarter.get("width", 512)),
            int(three_quarter.get("height", 512)),
        )
        w_3q = _smoothstep(0.16, 0.50, abs_x_norm) * (1.0 - 0.5 * _smoothstep(0.75, 1.0, abs_x_norm))
        merged[:, 2] = merged[:, 2] * (1.0 - 0.65 * w_3q) + pts_3q[:, 2] * (0.65 * w_3q)

    has_side = side is not None and bool(side.get("landmarks"))
    if has_side:
        pts_side, _ = canonicalize_view(
            side["landmarks"],
            side.get("matrix"),
            int(side.get("width", 512)),
            int(side.get("height", 512)),
        )
        # Outer jaw refinement from 3D canonicalized side view
        outer_jaw = _smoothstep(0.55, 0.90, abs_x_norm)
        w_jaw = np.clip(outer_jaw * 0.70, 0.0, 1.0)
        merged[:, 2] = merged[:, 2] * (1.0 - w_jaw) + pts_side[:, 2] * w_jaw

        # Direct 2D sagittal profile extraction matched by exact vertex ID (zero Y shift)
        merged = _apply_side_2d_profile(
            merged,
            neighbors,
            side["landmarks"],
            int(side.get("width", 512)),
            int(side.get("height", 512)),
        )

    if has_3q or has_side:
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
    }
