#!/bin/bash
# Plan-v2 Step 2 multi-seed + walking_xyz validation runner.
# - 4 walking_static seed runs (s44/s45 x d2_off/d2_b10)
# - walking_xyz vanilla (produces video.npz)
# - walking_xyz radseg_features precompute (~5 min)
# - walking_xyz d2_b10
#
# Outputs land in scene-specific dirs; ATE per run in
# Outputs/.../<scene>/traj/metrics_full_traj.txt and metrics_kf_traj.txt.

set -e
set -o pipefail

cd /home/cvg/HERMES-SLAM/DROID-W

PY=/home/cvg/miniconda3/envs/droid-w/bin/python
RADSEG_STATIC=/home/cvg/HERMES-SLAM/DROID-W/Outputs/TUM_RGBD/freiburg3_walking_static/radseg_features.npz

echo "[$(date +%H:%M:%S)] === Step 2 multi-seed batch begin ==="

# ── walking_static seeds 44 and 45, both d2_off and d2_b10 ──
for v in d2_off d2_b10; do
  for s in 44 45; do
    scene="freiburg3_walking_static_${v}_s${s}"
    cfg="configs/Dynamic/TUM_RGBD/${scene}.yaml"
    out="Outputs/TUM_RGBD/${scene}"
    rm -rf "$out"
    mkdir -p "$out"
    if [ "$v" = "d2_b10" ]; then
      ln -sf "$RADSEG_STATIC" "$out/radseg_features.npz"
    fi
    echo "[$(date +%H:%M:%S)] >>> launching $scene"
    $PY run.py --config "$cfg" > "/tmp/slamrun_${scene}.log" 2>&1
    echo "[$(date +%H:%M:%S)] <<< $scene complete"
  done
done

# ── walking_xyz d2_off (also produces video.npz used by precompute) ──
scene="freiburg3_walking_xyz_d2_off"
cfg="configs/Dynamic/TUM_RGBD/${scene}.yaml"
out="Outputs/TUM_RGBD/${scene}"
rm -rf "$out"
mkdir -p "$out"
echo "[$(date +%H:%M:%S)] >>> launching $scene (walking_xyz baseline)"
$PY run.py --config "$cfg" > "/tmp/slamrun_${scene}.log" 2>&1
echo "[$(date +%H:%M:%S)] <<< $scene complete"

# ── walking_xyz radseg precompute (uses video.npz from d2_off run above) ──
echo "[$(date +%H:%M:%S)] >>> radseg precompute on walking_xyz"
$PY scripts/precompute_radseg_features.py \
    --scene "Outputs/TUM_RGBD/freiburg3_walking_xyz_d2_off" \
    > /tmp/precompute_xyz.log 2>&1
# The precompute writes radseg_features.npz inside the --scene dir.
RADSEG_XYZ="${PWD}/Outputs/TUM_RGBD/freiburg3_walking_xyz_d2_off/radseg_features.npz"
if [ ! -f "$RADSEG_XYZ" ]; then
  echo "[ERROR] walking_xyz radseg precompute didn't produce $RADSEG_XYZ"
  tail -30 /tmp/precompute_xyz.log
  exit 2
fi
echo "[$(date +%H:%M:%S)] <<< radseg precompute complete"

# ── walking_xyz d2_b10 (symlinks the just-built radseg_features) ──
scene="freiburg3_walking_xyz_d2_b10"
cfg="configs/Dynamic/TUM_RGBD/${scene}.yaml"
out="Outputs/TUM_RGBD/${scene}"
rm -rf "$out"
mkdir -p "$out"
ln -sf "$RADSEG_XYZ" "$out/radseg_features.npz"
echo "[$(date +%H:%M:%S)] >>> launching $scene"
$PY run.py --config "$cfg" > "/tmp/slamrun_${scene}.log" 2>&1
echo "[$(date +%H:%M:%S)] <<< $scene complete"

echo "[$(date +%H:%M:%S)] === Step 2 multi-seed batch DONE ==="
echo ""
echo "=== ATE summary ==="
for scene in \
    freiburg3_walking_static_d2_off_s44 \
    freiburg3_walking_static_d2_off_s45 \
    freiburg3_walking_static_d2_b10_s44 \
    freiburg3_walking_static_d2_b10_s45 \
    freiburg3_walking_xyz_d2_off \
    freiburg3_walking_xyz_d2_b10; do
  full=$(grep -oE "'rmse': [0-9.]+" "Outputs/TUM_RGBD/$scene/traj/metrics_full_traj.txt" 2>/dev/null | head -1)
  kf=$(grep -oE "'rmse': [0-9.]+" "Outputs/TUM_RGBD/$scene/traj/metrics_kf_traj.txt" 2>/dev/null | head -1)
  printf "  %-40s  full=%s  kf=%s\n" "$scene" "$full" "$kf"
done
