"""
Scene-graph visualisation in rerun (post-processing).

Mirrors the entity layout used by DROID-W's live visualiser
(`src/utils/droid_visualization_rerun.py`):

    world
    ├── cameras/<kf_idx>           (camera pose + frustum, like DROID-W)
    └── points/<kf_idx>            (per-keyframe colored point cloud)

Difference from DROID-W's live stream: each 3D point is coloured by its
**instance ID** (from the FastSAM/CLIP/SigLIP-2 mask track) instead of its
RGB. Pixels that fall inside a mask classified as DYNAMIC are forced to a
bright red so they pop visually.

Designed for post-processing — accepts arrays read from `video.npz` plus the
`all_detections` / `tracks` produced by the perception pipeline. No
dependency on the running SLAM process or its CUDA backends.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import rerun as rr


# -----------------------------------------------------------------------------
# Pose helpers (vendored small to avoid pulling seg_model into this module)
# -----------------------------------------------------------------------------

def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (DROID-W TUM convention) to a 3x3 rotation."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),     s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [    s*(qx*qy + qz*qw), 1 - s*(qx*qx + qz*qz),     s*(qy*qz - qx*qw)],
        [    s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _pose_to_world_from_cam(pose_7: Sequence[float]) -> np.ndarray:
    """tum_poses row → 4x4 camera-to-world matrix (same convention DROID-W uses)."""
    tx, ty, tz, qx, qy, qz, qw = pose_7
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T[:3,  3] = (tx, ty, tz)
    return T


# -----------------------------------------------------------------------------
# Color helpers
# -----------------------------------------------------------------------------

def _track_color(track_id: int) -> np.ndarray:
    """Deterministic, well-spread RGB colour for a given track ID.

    Uses a golden-ratio hash on the integer ID so consecutive IDs land far
    apart in hue space. Returns uint8 (3,).
    """
    if track_id < 0:
        return np.array([180, 180, 180], dtype=np.uint8)
    # Golden-ratio hue, full saturation, fixed value.
    h = (track_id * 0.6180339887498949) % 1.0
    s, v = 0.78, 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = table[i % 6]
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


DYNAMIC_RGB = np.array([220, 30, 30], dtype=np.uint8)
UNMASKED_RGB_DIM = 0.55  # darken bare RGB for un-instanced pixels


# -----------------------------------------------------------------------------
# Frustum (matches DROID-W _camera_lines style)
# -----------------------------------------------------------------------------

def _camera_frustum(scale: float = 0.1) -> np.ndarray:
    """Returns a (S, 2, 3) array of line segments forming a camera frustum
    in camera-local coordinates (looking down +Z, OpenCV RDF axes)."""
    s = scale
    apex = np.array([0.0, 0.0, 0.0])
    far_z = s * 2.0
    far_w = s * 1.5
    far_h = s * 1.0
    corners = np.array([
        [-far_w, -far_h, far_z],
        [ far_w, -far_h, far_z],
        [ far_w,  far_h, far_z],
        [-far_w,  far_h, far_z],
    ])
    segs = []
    for c in corners:
        segs.append(np.stack([apex, c]))
    for i in range(4):
        segs.append(np.stack([corners[i], corners[(i + 1) % 4]]))
    return np.stack(segs, axis=0).astype(np.float32)  # (8, 2, 3)


# -----------------------------------------------------------------------------
# Per-frame instance map
# -----------------------------------------------------------------------------

def _build_instance_map(
    detections: List[Dict],
    H: int,
    W: int,
    tracks: Optional[Dict[int, Dict]] = None,
) -> tuple:
    """Rasterise per-pixel instance IDs and dynamic flags from detections.

    Detections drawn in confidence-ascending order so the highest-confidence
    mask wins on overlap. Returns:
        instance_id : (H, W) int32, -1 where no mask covers the pixel
        is_dynamic  : (H, W) bool, True where the winning mask is dynamic
    """
    instance_id = np.full((H, W), -1, dtype=np.int32)
    is_dynamic = np.zeros((H, W), dtype=bool)
    if not detections:
        return instance_id, is_dynamic

    # Lower confidence first so high-confidence masks overwrite.
    dets_sorted = sorted(detections, key=lambda d: d.get("confidence", 0.0))
    for d in dets_sorted:
        m = d.get("mask")
        if m is None:
            continue
        tid = int(d.get("track_id", -1))
        if tid < 0:
            continue
        if m.shape != (H, W):
            # Skip rather than guess — caller is expected to pass image-res masks
            continue
        # Track-level dynamic flag (preferred) → fall back to per-detection flag.
        dyn = False
        if tracks is not None and tid in tracks:
            dyn = bool(tracks[tid].get("is_dynamic", False))
        else:
            dyn = bool(d.get("is_dynamic", False))
        sel = m > 0
        instance_id[sel] = tid
        is_dynamic[sel] = dyn
    return instance_id, is_dynamic


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def stream_scene_graph(
    images: np.ndarray,
    poses: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    all_detections: List[List[Dict]],
    tracks: Optional[Dict[int, Dict]] = None,
    app_id: str = "hermes-scene-graph",
    record_path: Optional[str] = None,
    spawn: bool = True,
    web_port: int = 9876,
    point_subsample: int = 4,
    min_depth: float = 0.05,
    max_depth: float = 25.0,
    frustum_scale: float = 0.1,
) -> None:
    """Stream a coloured-by-instance-ID 3D reconstruction to rerun.

    Args:
        images:         (N, 3, H, W) float32 in [0, 1] or uint8.
        poses:          (N, 7) tum_poses [tx,ty,tz,qx,qy,qz,qw] (DROID-W format).
        depths:         (N, H, W) float — metric depth at image resolution.
        K:              (3, 3) intrinsics matrix at image resolution.
        all_detections: per-keyframe lists of detection dicts. Each must carry
                        'mask' (image-res uint8), 'track_id', 'is_dynamic',
                        'confidence'.
        tracks:         optional {track_id -> {label, is_dynamic, ...}}.
        app_id:         rerun application id.
        record_path:    optional .rrd file path to record alongside the stream.
        spawn:          if True, spawn the native rerun viewer.
        web_port:       fallback web-viewer port if native spawn fails.
        point_subsample: stride applied to the image grid before unprojection.
                         4 → ~1/16 of pixels; keeps the viewer responsive.
        min_depth/max_depth: clip bad depths.
        frustum_scale:  size of camera frustum lines.
    """
    N = len(images)
    if N == 0:
        return
    _, _, H, W = images.shape if images.ndim == 4 else (1, 3, *images.shape[-2:])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Convert images to a uniform (N, H, W, 3) uint8 format for logging.
    if images.dtype != np.uint8:
        if images.max() <= 1.5:
            imgs_u8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
        else:
            imgs_u8 = np.clip(images, 0, 255).astype(np.uint8)
    else:
        imgs_u8 = images
    if imgs_u8.shape[1] == 3:
        imgs_u8 = imgs_u8.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    # Subsampled pixel grid (shared across all keyframes)
    s = max(1, int(point_subsample))
    vs, us = np.meshgrid(
        np.arange(0, H, s, dtype=np.int32),
        np.arange(0, W, s, dtype=np.int32),
        indexing="ij",
    )
    grid_h, grid_w = vs.shape

    # rerun init + viewer
    try:
        rr.init(app_id, spawn=False)
    except TypeError:
        rr.init(app_id)
    started = False
    if spawn:
        try:
            rr.spawn()
            started = True
            print("[Rerun] Native viewer spawned")
        except Exception:
            pass
        if not started:
            try:
                rr.serve_web_viewer(web_port=web_port, open_browser=True)
                started = True
                print(f"[Rerun] Web viewer on http://127.0.0.1:{web_port}")
            except Exception:
                print("[Rerun] No live viewer available")
    if record_path:
        try:
            rr.save(record_path)
            print(f"[Rerun] Recording to {record_path}")
        except Exception as e:
            print(f"[Rerun] Recording failed: {e}")

    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
    except Exception:
        pass

    frustum_local = _camera_frustum(scale=frustum_scale)  # (S, 2, 3)

    # Per-keyframe logging
    for kf in range(N):
        try:
            rr.set_time_sequence("keyframe", kf)
        except Exception:
            pass
        T_w_c = _pose_to_world_from_cam(poses[kf])  # 4x4 cam-to-world
        cam_path = f"world/cameras/{kf:06d}"

        # Pose
        try:
            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=T_w_c[:3, 3].astype(np.float32),
                    mat3x3=T_w_c[:3, :3].astype(np.float32),
                ),
            )
        except Exception:
            pass

        # Frustum lines (transform local segments into world)
        try:
            segs_world = []
            for seg in frustum_local:
                seg_h = np.concatenate([seg, np.ones((2, 1), dtype=np.float32)], axis=1)
                seg_w = (T_w_c @ seg_h.T).T[:, :3]
                segs_world.append(seg_w)
            rr.log(f"{cam_path}/frustum", rr.LineStrips3D(np.stack(segs_world, axis=0)))
        except Exception:
            pass

        # Build per-pixel instance + dynamic maps
        dets = all_detections[kf] if kf < len(all_detections) else []
        inst_map, dyn_map = _build_instance_map(dets, H, W, tracks)

        # Sample depth at the subsampled grid
        d_grid = depths[kf][vs, us].astype(np.float32)
        u_n = (us - cx) / fx
        v_n = (vs - cy) / fy
        cam_xyz = np.stack([u_n * d_grid, v_n * d_grid, d_grid], axis=-1)  # (h, w, 3)
        cam_xyz_h = np.concatenate(
            [cam_xyz, np.ones((grid_h, grid_w, 1), dtype=np.float32)], axis=-1
        ).reshape(-1, 4)
        world_xyz = (T_w_c @ cam_xyz_h.T).T[:, :3]

        # Sample masks at the same subsampled grid
        inst_grid = inst_map[vs, us]
        dyn_grid = dyn_map[vs, us]
        rgb_grid = imgs_u8[kf][vs, us]                       # (h, w, 3)

        # Validity (depth + finite + bounded)
        valid = (
            (d_grid >= min_depth)
            & (d_grid <= max_depth)
            & np.isfinite(d_grid)
        )
        valid_flat = valid.reshape(-1)
        if not valid_flat.any():
            continue
        pts = world_xyz[valid_flat]
        inst_flat = inst_grid.reshape(-1)[valid_flat]
        dyn_flat = dyn_grid.reshape(-1)[valid_flat]
        rgb_flat = rgb_grid.reshape(-1, 3)[valid_flat].astype(np.uint8)

        # Compute per-point colour
        colors = (rgb_flat.astype(np.float32) * UNMASKED_RGB_DIM).astype(np.uint8)
        # Instance-coloured pixels (replace dim RGB with track colour)
        instanced = inst_flat >= 0
        if instanced.any():
            unique_ids = np.unique(inst_flat[instanced])
            for tid in unique_ids:
                if tid < 0:
                    continue
                sel = inst_flat == tid
                colors[sel] = _track_color(int(tid))
        # Dynamic pixels overwrite to red
        if dyn_flat.any():
            colors[dyn_flat] = DYNAMIC_RGB

        try:
            rr.log(f"world/points/{kf:06d}", rr.Points3D(pts.astype(np.float32), colors=colors))
        except Exception:
            try:
                rr.log(f"world/points/{kf:06d}", rr.Points3D(pts.astype(np.float32)))
            except Exception:
                pass

    print(
        f"[scene_graph_rerun] Logged {N} keyframes "
        f"(grid {grid_h}x{grid_w} per frame, subsample={s})"
    )
