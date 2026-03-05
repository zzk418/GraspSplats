#!/usr/bin/env python3
"""
Convert COLMAP images.txt to per-frame poses/*.npy files,
matching the format of scene_data/example_data/poses/.

Each .npy file is a 4x4 float64 camera-to-world (c2w) transformation matrix.
COLMAP stores world-to-camera (w2c) as quaternion + translation,
so we invert it here.

Usage:
    python scripts/colmap_to_poses.py <scene_dir>

Example:
    python scripts/colmap_to_poses.py scene_data/tissue_data
"""

import sys
import os
import numpy as np
from pathlib import Path


def quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def w2c_to_c2w(R, t):
    """Convert world-to-camera (R, t) to 4x4 camera-to-world matrix."""
    c2w = np.eye(4, dtype=np.float64)
    # c2w rotation = R^T, translation = -R^T @ t
    c2w[:3, :3] = R.T
    c2w[:3,  3] = -R.T @ t
    return c2w


def parse_images_txt(images_txt_path):
    """Parse COLMAP images.txt, return dict: image_name -> 4x4 c2w matrix."""
    poses = {}
    with open(images_txt_path, "r") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    # images.txt has 2 lines per image: pose line + points2D line
    for i in range(0, len(lines), 2):
        tokens = lines[i].split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
        tx,  ty,  tz   = float(tokens[5]), float(tokens[6]), float(tokens[7])
        name = tokens[9]

        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        c2w = w2c_to_c2w(R, t)
        poses[name] = c2w

    return poses


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    scene_dir = Path(sys.argv[1])
    images_txt = scene_dir / "sparse" / "0" / "images.txt"
    if not images_txt.exists():
        # fallback: sparse/images.txt
        images_txt = scene_dir / "sparse" / "images.txt"
    if not images_txt.exists():
        print(f"[ERROR] images.txt not found in {scene_dir}/sparse/0/ or {scene_dir}/sparse/")
        sys.exit(1)

    poses_dir = scene_dir / "poses"
    poses_dir.mkdir(exist_ok=True)

    print(f"Reading: {images_txt}")
    poses = parse_images_txt(images_txt)

    saved = 0
    for name, c2w in poses.items():
        # derive index from filename stem (e.g. "5.png" -> 5)
        stem = Path(name).stem
        out_path = poses_dir / f"{stem}.npy"
        np.save(out_path, c2w)
        saved += 1

    print(f"Saved {saved} pose files -> {poses_dir}")


if __name__ == "__main__":
    main()
