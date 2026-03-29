import os
import time
import torch
import numpy as np
import rerun as rr
from lietorch import SE3
import droid_backends
import imageio.v2 as imageio
import shutil
import subprocess

# --- headless safety (no Qt/GUI on cluster) ---
# Only set offscreen Qt if no display is available
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ---- helpers to draw a camera frustum as 3D line segments ----
_CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [ 1, -1, 1.5],
    [ 1,  1, 1.5],
    [-1,  1, 1.5],
    [-0.5, 1, 1.5],
    [ 0.5, 1, 1.5],
    [ 0, 1.2, 1.5]
], dtype=np.float32)

_CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]
], dtype=np.int32)

def _camera_lines(scale=0.05):
    pts = (_CAM_POINTS * scale).astype(np.float32)
    return [np.stack([pts[a], pts[b]], axis=0) for a, b in _CAM_LINES]  # list[(2,3)]

def _to_rr_transform(mat4):
    mat4 = mat4.astype(np.float32)
    try:
        return rr.Transform3D(mat4x4=mat4)   # newer API
    except TypeError:
        return rr.Transform3D(mat4)          # older positional form

def _rr_init(app_id: str):
    # rerun 0.30+: rr.init() takes application_id
    for kwargs in [
        dict(application_id=app_id),
        dict(app_id=app_id),
        dict(),
    ]:
        try:
            rr.init(**kwargs)
            return
        except TypeError:
            continue
    # positional fallback
    try:
        rr.init(app_id)
    except Exception:
        rr.init()

def droid_visualization_rerun(
    video,
    device="cuda:0",
    app_id="droid_viz",
    web_port=9876,                # web server port on the node
    record_path=None,             # e.g., "output_fastlivo/debug_rerun/stream.rrd"
    warmup=8,
    filter_thresh=0.005, #filter_thresh=0.005,
    scale=1.0,
    exit_event=None,
):
    """
    Start a headless Rerun Web Viewer on the SLURM node and stream data to it.
    View remotely by SSH port-forwarding to your laptop:
        ssh -N -L 9876:localhost:9876 zihzhu@eu-g4-020.euler.ethz.ch
    Then open http://localhost:9876 in your laptop browser.
    """
    torch.cuda.set_device(device)
    _rr_init(app_id)

    # Try native viewer first, fall back to web viewer, then file-only
    started = False
    try:
        rr.spawn()
        started = True
        print("[Rerun] Native viewer spawned and connected")
    except Exception:
        pass
    if not started:
        try:
            rr.serve_web_viewer(web_port=web_port, open_browser=True)
            started = True
            print(f"[Rerun] Web viewer on http://127.0.0.1:{web_port}")
        except Exception:
            pass
    if not started and not record_path:
        print("[Rerun] No viewer available and no record path set")

    # Always save to file if record_path is set (in addition to live viewer)
    if record_path:
        try:
            rr.save(record_path)
            print(f"[Rerun] Recording to {record_path}")
        except Exception as e:
            print(f"[Rerun] Recording not started: {e}")

    # Global scene axes (best-effort across versions)
    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)  # OpenCV-ish axes
    except Exception:
        pass

    filter_thresh = float(filter_thresh)
    cam_segs_local = _camera_lines(scale=0.1)  # larger for current cam

    try:
        while True:
            # Check which frames need updating
            with video.get_lock():
                dirty_index, = torch.where(video.dirty.clone())
            
            if exit_event.is_set() and len(dirty_index) == 0:
                print("Final clean, exiting...")
                break

            elif len(dirty_index) == 0:
                time.sleep(0.01)
                continue

            # Mark processed
            video.dirty[dirty_index] = False

            high_res = True
            # Fetch tensors
            poses = torch.index_select(video.poses, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()  # [N, 4, 4] world_T_cam
            images = torch.index_select(video.images, 0, dirty_index.cpu())
            intrinsics = video.intrinsics[0].clone()
            if high_res:
                # disps = torch.index_select(video.disps_up, 0, dirty_index.cpu())
                disps = torch.index_select(video.disps_up, 0, dirty_index)
                # disps = disps.cpu()
                images = images.cpu()[..., :, :].permute(0, 2, 3, 1)  # [N, H, W, 3]
                intrinsics*=8
            else:
                disps = torch.index_select(video.disps, 0, dirty_index)
                images = images.cpu()[..., 3::8, 3::8].permute(0, 2, 3, 1)  # [N, H, W, 3]
            # disps_up = torch.index_select(video.disps_up, 0, dirty_index)

            # Consistency filter
            thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            if high_res:
                # disps_up_clone = video.disps_up.clone()
                count = droid_backends.depth_filter(video.poses, video.disps_up, intrinsics, dirty_index, thresh)
            else:
                count = droid_backends.depth_filter(video.poses, video.disps, intrinsics, dirty_index, thresh)
            # print(count)
            count = count.cpu()
            disps = disps.cpu()
            
            masks = ((count >= 2) & (disps > 0.25 * disps.mean(dim=[1, 2], keepdim=True)))  # [N,H,W]

            points = droid_backends.iproj(SE3(poses).inv().data, disps.cuda(), intrinsics).cpu()  # [N,H,W,3]

            
            # Log per keyframe
            for i in range(len(dirty_index)):
                kf_idx = int(dirty_index[i].item())
                world_from_cam = Ps[i]
                cam_path = f"world/cameras/{kf_idx:06d}"

                # Pose
                try:
                    rr.log(cam_path, _to_rr_transform(world_from_cam))
                except Exception:
                    pass

                # Frustum lines
                try:
                    segs_world = []
                    for seg in cam_segs_local:
                        seg_h = np.concatenate([seg, np.ones((2, 1), dtype=np.float32)], axis=1)  # (2,4)
                        seg_w = (world_from_cam @ seg_h.T).T[:, :3]
                        segs_world.append(seg_w)
                    rr.log(f"{cam_path}/frustum", rr.LineStrips3D(np.stack(segs_world, axis=0)))  # (S,2,3)
                except Exception:
                    pass

                # Colored point cloud
                mask = masks[i].reshape(-1).numpy()
                pts = points[i].reshape(-1, 3)[mask].numpy()
                clr = images[i].reshape(-1, 3)[mask].numpy()
                if pts.size > 0:
                    try:
                        rr.log(f"world/points/{kf_idx:06d}", rr.Points3D(pts, colors=(clr * 255).astype(np.uint8)))
                    except Exception:
                        rr.log(f"world/points/{kf_idx:06d}", rr.Points3D(pts))

    except KeyboardInterrupt:
        pass
