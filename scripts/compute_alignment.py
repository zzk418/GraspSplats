"""
Compute world2base transform for a scene with no robot calibration.

Strategy:
  1. Rotation:
     - robot Z (up): estimated from point cloud normal (PC2, least variance axis),
       sign corrected so cameras are above the table (camera centers have positive Z).
     - robot Y (long edge): PC0 of point cloud projected onto the table plane.
     - robot X: cross(Y, Z)
  2. Scale: COLMAP scene long-edge -> real object size (meters)
  3. Translation: scene center -> target_pos in robot base frame

Requires: colmap images.bin in same sparse/0 folder as points3D.ply

Usage:
    python scripts/compute_alignment.py \
        --ply scene_data/tissue_data/colmap/sparse/0/points3D.ply \
        --real_size 1 \
        --target_pos 0.5 0.0 0.05 \
        --out outputs/tissue_data/world2base.npy
"""

import argparse
import sys
import numpy as np
from plyfile import PlyData

sys.path.append("./feature-splatting-inria")
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat


def compute_world2base(ply_path: str, real_size: float, target_pos: np.ndarray) -> np.ndarray:
    ply = PlyData.read(ply_path)
    pts = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)

    center = pts.mean(axis=0)
    span = (pts.max(axis=0) - pts.min(axis=0)).max()
    scale = real_size / span
    print(f"COLMAP span: {span:.4f} units  |  scale: {scale:.6f} m/unit")

    # --- rotation ---
    # SVD of centered point cloud: rows of Vt are principal axes
    # PC0 = longest axis (table plane), PC2 = normal to table (up/down)
    centered = pts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    scene_up = Vt[2]  # least-variance axis = table normal

    # Sign correction: camera centers should be on the positive-Z side of the table.
    # Average camera center in COLMAP coords:
    sparse_dir = ply_path.replace("points3D.ply", "")
    extrinsics = read_extrinsics_binary(sparse_dir + "images.bin")
    cam_centers = []
    for v in extrinsics.values():
        R_c = qvec2rotmat(v.qvec)
        t_c = np.array(v.tvec)
        cam_centers.append(-R_c.T @ t_c)
    mean_cam = np.mean(cam_centers, axis=0) - center
    if np.dot(scene_up, mean_cam) < 0:
        scene_up = -scene_up
    print(f"Table normal (scene up): {scene_up}")

    # robot Y (long edge) = PC0 projected onto table plane
    scene_long = Vt[0]
    scene_long = scene_long - np.dot(scene_long, scene_up) * scene_up
    scene_long /= np.linalg.norm(scene_long)

    # robot X = cross(Y, Z), right-handed
    scene_x = np.cross(scene_long, scene_up)
    scene_x /= np.linalg.norm(scene_x)

    # R rows: colmap axes mapped to robot axes
    R = np.stack([scene_x, scene_long, scene_up], axis=0)
    if np.linalg.det(R) < 0:
        R[0] = -R[0]
    print(f"Rotation det: {np.linalg.det(R):.4f}")

    # world2base: p_robot = scale * R @ p_colmap + t
    t = np.array(target_pos) - scale * R @ center
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = t
    return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--real_size", type=float, default=0.05,
                        help="Real-world long-edge size of the object in meters")
    parser.add_argument("--target_pos", type=float, nargs=3, default=[0.5, 0.0, 0.05],
                        help="Scene center in robot base frame (x y z meters)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    M = compute_world2base(args.ply, args.real_size, np.array(args.target_pos))
    print("\nworld2base =\n", M)
    np.save(args.out, M)
    print(f"Saved to {args.out}")
