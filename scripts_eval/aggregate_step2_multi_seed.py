"""Aggregate Step 2 multi-seed ATEs into a paired-delta summary.

Reads `Outputs/TUM_RGBD/<scene>/traj/metrics_{full,kf}_traj.txt` for each
configured scene, parses the `rmse` field, and emits:

- a markdown table of per-seed full/KF ATEs,
- per-seed paired deltas (d2_b10 - d2_off) at matching seeds,
- mean +/- std over seeds for each metric,
- one-line summary verdict.

Usage (cvg):
    python /tmp/step2_aggregate.py
"""
from __future__ import annotations
import os, re, math
from pathlib import Path

ROOT = Path("/home/cvg/HERMES-SLAM/DROID-W/Outputs/TUM_RGBD")

# (scene-name, seed, condition) tuples we expect to find.
WALKING_STATIC = [
    ("freiburg3_walking_static_d2_off",     43, "off"),  # from earlier Step 2 batch
    ("freiburg3_walking_static_d2_off_s44", 44, "off"),
    ("freiburg3_walking_static_d2_off_s45", 45, "off"),
    ("freiburg3_walking_static_d2_b10",     43, "b10"),  # from earlier Step 2 batch
    ("freiburg3_walking_static_d2_b10_s44", 44, "b10"),
    ("freiburg3_walking_static_d2_b10_s45", 45, "b10"),
]
WALKING_XYZ = [
    ("freiburg3_walking_xyz_d2_off", 43, "off"),
    ("freiburg3_walking_xyz_d2_b10", 43, "b10"),
]

RX = re.compile(r"'rmse':\s*([0-9.eE+-]+)")


def read_rmse(scene: str, traj_kind: str) -> float | None:
    p = ROOT / scene / "traj" / f"metrics_{traj_kind}_traj.txt"
    if not p.exists():
        return None
    m = RX.search(p.read_text())
    if not m:
        return None
    return float(m.group(1)) * 1000.0  # m -> mm


def fmt(x):
    return f"{x:.3f}" if x is not None else "  n/a"


def stats(xs):
    n = len(xs)
    if n == 0:
        return None, None
    mu = sum(xs) / n
    if n < 2:
        return mu, None
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def section(title: str, rows):
    print(f"\n## {title}\n")
    print(f"| scene | seed | cond | full [mm] | kf [mm] |")
    print(f"|---|---:|---|---:|---:|")
    data = {}
    for scene, seed, cond in rows:
        full = read_rmse(scene, "full")
        kf = read_rmse(scene, "kf")
        print(f"| `{scene}` | {seed} | {cond} | {fmt(full)} | {fmt(kf)} |")
        data.setdefault(cond, {})[seed] = (full, kf)

    # Per-seed paired delta (b10 - off)
    if "off" in data and "b10" in data:
        print(f"\n**paired deltas (b10 - off) [mm]:**\n")
        print(f"| seed | dfull | dkf |")
        print(f"|---:|---:|---:|")
        d_full_list, d_kf_list = [], []
        for s in sorted(set(data["off"]) & set(data["b10"])):
            o = data["off"][s]; b = data["b10"][s]
            if o[0] is not None and b[0] is not None:
                df = b[0] - o[0]; d_full_list.append(df)
            else: df = None
            if o[1] is not None and b[1] is not None:
                dk = b[1] - o[1]; d_kf_list.append(dk)
            else: dk = None
            print(f"| {s} | {fmt(df)} | {fmt(dk)} |")
        mu_full, sd_full = stats(d_full_list)
        mu_kf, sd_kf = stats(d_kf_list)
        print(f"\n**mean +/- std (n={len(d_full_list)}):** "
              f"d_full = {fmt(mu_full)} +/- {fmt(sd_full)} mm  ;  "
              f"d_kf = {fmt(mu_kf)} +/- {fmt(sd_kf)} mm")
        # Verdict
        verdict = "POSITIVE" if (mu_full is not None and mu_full < 0) else "REGRESSIVE / NEUTRAL"
        print(f"\n**direction:** {verdict}  "
              f"(negative d_full means D.2 reduces full-trajectory ATE)")


def main():
    section("walking_static (3 seeds)", WALKING_STATIC)
    section("walking_xyz (1 seed)", WALKING_XYZ)


if __name__ == "__main__":
    main()
