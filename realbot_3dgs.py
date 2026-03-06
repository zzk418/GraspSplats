"""
realbot_3dgs.py — GraspSplats interactive 3DGS viewer with click-to-grasp

Key design:
  - nerfstudio-style per-client GaussianRenderThread with adaptive resolution
    (low_move → low_static → high, triggered by camera.on_update)
  - Accurate click-to-segment using render_gaussian_idx:
      1. On click, re-render a full-res frame with render_gaussian_idx=True
      2. Project the click ray to screen UV → look up the rendered Gaussian index
      3. Spatial flood-fill + DBSCAN from that seed Gaussian
  - Automatic pipeline: segment → GPD grasps → IK reachability → best grasp

Usage:
    python realbot_3dgs.py -m outputs/tissue_data
"""

import sys, os, time, copy, threading
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append("./feature-splatting-inria")

import numpy as np
import torch
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
import transforms3d.euler as euler
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import viser
import viser.transforms as vtf
from viser.extras import ViserUrdf

from scene import Scene, skip_feat_decoder
from scene.cameras import MiniCam
from arguments import ModelParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, render
from grasping import grasping_utils, plan_utils
from gaussian_edit import edit_utils
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.sh_utils import eval_sh


# ── Camera convention ─────────────────────────────────────────────────────────
# viser:  OpenCV  (+Z forward, -Y up)
# 3DGS:   OpenGL  (+Z backward, +Y up)
# nerfstudio applies SO3.from_x_radians(pi) (flip Y,Z cols) to convert.
# We do the same then pass to getWorld2View2.

def viser_to_minicam(client: viser.ClientHandle, width: int, height: int,
                     base2colmap: np.ndarray = None) -> MiniCam:
    """
    Build a 3DGS MiniCam from the viser client camera.

    viser scene = COLMAP frame (Gaussians live here directly).
    No coordinate transform needed — just convert camera convention:
      viser OpenCV (+Z fwd, -Y up) → 3DGS OpenGL (+Z back, +Y up)
    """
    # viser camera: R_wc in viser/OpenCV convention
    R_wc_cv = vtf.SO3(np.asarray(client.camera.wxyz)).as_matrix()
    # flip Y,Z cols: OpenCV → OpenGL/3DGS convention
    flip = np.diag([1.0, -1.0, -1.0])
    R_wc = R_wc_cv @ flip                        # world(COLMAP)←camera
    pos  = np.asarray(client.camera.position, dtype=np.float64)

    # getWorld2View2(R, t) stores R^T internally, so pass R_wc (not R_cw)
    # t_cw = -R_cw @ pos = -(R_wc^T) @ pos
    R_cw = R_wc.T
    t_cw = (-R_cw @ pos).astype(np.float32)

    world_view = torch.tensor(
        getWorld2View2(R_wc.astype(np.float32), t_cw)
    ).transpose(0, 1).cuda()

    fovy = float(client.camera.fov)
    fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * client.camera.aspect)

    proj = getProjectionMatrix(znear=0.001, zfar=100.0,
                               fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, 0.001, 100.0, world_view, full_proj)


# ── Render State Machine ──────────────────────────────────────────────────────
# Borrowed from nerfstudio/viewer/render_state_machine.py

_RES = {"low_move": (400, 300), "low_static": (640, 480), "high": (960, 720)}
_TRANS = {
    "low_move":   {"move": "low_move",   "static": "low_static", "rerender": "low_static"},
    "low_static": {"move": "low_move",   "static": "high",       "rerender": "low_static"},
    "high":       {"move": "low_move",   "static": "high",       "rerender": "low_static"},
}


class GaussianRenderThread(threading.Thread):
    def __init__(self, client: viser.ClientHandle, renderer: "GaussianRenderer",
                 base2colmap: np.ndarray = None):
        super().__init__(daemon=True)
        self.client      = client
        self.renderer    = renderer
        self.base2colmap = base2colmap
        self.state    = "low_static"
        self._next: Optional[str] = None
        self._lock    = threading.Lock()
        self._trigger = threading.Event()
        self.running  = True

    def action(self, act: str):
        with self._lock:
            if self._next != "rerender":   # rerender is never overwritten
                self._next = act
        self._trigger.set()

    def run(self):
        while self.running:
            if not self._trigger.wait(timeout=0.25):
                self.action("static")
                continue
            self._trigger.clear()
            with self._lock:
                act, self._next = self._next, None
            if act is None:
                continue
            if self.state == "high" and act == "static":
                continue
            self.state = _TRANS[self.state][act]
            w, h = _RES[self.state]
            try:
                mini_cam = viser_to_minicam(self.client, w, h, self.base2colmap)
                img = self.renderer.render_rgb(mini_cam, w, h)
                # Debug: print camera center on first render
                if not getattr(self, '_first_rendered', False):
                    cc = mini_cam.camera_center.cpu().numpy()
                    print(f"[RenderThread] first render cam_center={cc}, "
                          f"pos_base={self.client.camera.position}")
                    self._first_rendered = True
                quality = 80 if self.state != "high" else 92
                self.client.set_background_image(img, format="jpeg",
                                                 jpeg_quality=quality)
            except Exception as e:
                import traceback
                print(f"[RenderThread] error: {e}")
                traceback.print_exc()


class GaussianRenderer:
    """Thread-safe 3DGS renderer with optional highlight overlay."""

    def __init__(self, gaussians: GaussianModel, pipe):
        self.gaussians = gaussians
        self.pipe      = pipe
        self._lock     = threading.Lock()
        self.highlight_mask: Optional[np.ndarray] = None  # bool (N,)

    def set_highlight(self, mask: Optional[np.ndarray]):
        with self._lock:
            self.highlight_mask = None if mask is None else mask.copy()

    def render_rgb(self, mini_cam: MiniCam, width: int, height: int) -> np.ndarray:
        """Render to uint8 H×W×3 with optional red highlight."""
        import cv2
        bg  = torch.zeros(3, device="cuda", dtype=torch.float32)
        gs  = self.gaussians
        with self._lock:
            hmask = self.highlight_mask

        with torch.no_grad():
            if hmask is not None:
                N = gs.get_xyz.shape[0]
                shs_v = gs.get_features.transpose(1, 2).view(
                    -1, 3, (gs.max_sh_degree + 1) ** 2)
                d = gs.get_xyz - mini_cam.camera_center.repeat(N, 1)
                d = d / (d.norm(dim=1, keepdim=True) + 1e-6)
                rgb = torch.clamp_min(eval_sh(gs.active_sh_degree, shs_v, d) + 0.5, 0.0)
                colors        = rgb * 0.25
                colors[hmask] = torch.tensor([1.0, 0.25, 0.25], device="cuda")
                out = render(mini_cam, gs, self.pipe, bg, override_color=colors)
            else:
                out = render(mini_cam, gs, self.pipe, bg)

        img = out["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        return img

    def render_gaussian_idx_map(self, mini_cam: MiniCam) -> torch.Tensor:
        """
        Render a frame with render_gaussian_idx=True.
        Returns H×W×N int32 tensor (N = max Gaussians per pixel, -1 = empty).
        """
        bg = torch.zeros(3, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            out = render(mini_cam, self.gaussians, self.pipe, bg,
                         render_gaussian_idx=True)
        return out["render_gaussian_idx"]   # H×W×N


# ── Click → Gaussian index via render_gaussian_idx ────────────────────────────

def click_to_gaussian_idx(client: viser.ClientHandle,
                           renderer: GaussianRenderer,
                           render_w: int = 960, render_h: int = 720) -> int:
    """
    Re-render a full-res frame with render_gaussian_idx=True and look up the
    Gaussian under the click pixel.

    The click gives us a ray (origin, direction) in world space.  We project
    that ray back to screen UV by calling viser_to_minicam and doing the
    standard perspective projection.

    Returns Gaussian index, or -1 if the pixel is background.
    """
    mini_cam = viser_to_minicam(client, render_w, render_h)
    idx_map  = renderer.render_gaussian_idx_map(mini_cam)   # H×W×N

    # Get the camera intrinsics from mini_cam
    fovx = mini_cam.FoVx
    fovy = mini_cam.FoVy
    fx   = render_w / (2.0 * np.tan(fovx * 0.5))
    fy   = render_h / (2.0 * np.tan(fovy * 0.5))
    cx, cy = render_w / 2.0, render_h / 2.0

    # camera center in world
    cam_pos = mini_cam.camera_center.cpu().numpy()   # (3,)

    # We need the ray direction in *camera* space.
    # In viser, click gives ray_direction in world space.
    # We can instead recover the screen UV from the world2view matrix:
    #   p_cam = R_cw @ (ray_origin - cam_pos) + t_cw
    # Since ray_origin == cam_pos for a camera ray, we use the direction.
    # world_view is column-major: p_cam = world_view^T * p_world_h (OpenGL convention)

    # Convert direction from world to camera space
    W2V = mini_cam.world_view_transform.cpu().numpy()  # 4×4, column-major
    # p_cam = W2V^T @ p_world_h  (but for direction, drop translation)
    R_cw = W2V[:3, :3].T   # rotation: world→cam, OpenGL convention
    # In 3DGS world view: column j of W2V is the j-th world-space basis in camera space
    # Actually world_view = getWorld2View2(R_cw, t_cw).T  so rows are camera axes in world
    # p_cam = W2V.T @ [x,y,z,1]
    # For a direction: dir_cam = W2V[:3,:3].T @ dir_world
    #                          = (transposed 3×3 of column-major world_view) @ dir_world

    # ray_direction from viser (world space, OpenCV +Z forward, -Y up)
    # but our MiniCam uses OpenGL, so we must undo the flip we applied
    # Actually: the render and the click share the same world frame, so we just need
    # to project via the same projection we used for rendering.

    # NDC from clip: x_ndc = (p_cam.x / -p_cam.z) * fx / (W/2) ... simpler:
    # screen_u = fx * (dir_cam.x / dir_cam.z_forward) + cx
    # In OpenGL camera: +Z is *backward* so forward = -Z
    # So: screen_u = fx * (-dir_cam.x / dir_cam.z) + cx  ... but 3DGS uses its own projection
    # Let's just use the standard pinhole formula with the projection matrix.

    # Use the 3DGS full_proj_transform to project a world point along the ray.
    # We project cam_pos + ray_dir (1 unit away from camera).
    ray_world = np.asarray(client.camera.wxyz)  # placeholder — we'll use camera.position + ray_dir

    # Simplest: project a point 1 unit along the click ray in world space
    ray_d_world = np.array(client.camera.wxyz)  # This is wrong; we need the actual ray_direction
    # We don't have ray_direction here (it's only in ScenePointerEvent).
    # So we can't reconstruct UV without it. Caller must pass it in.
    raise RuntimeError("Use click_to_gaussian_idx_from_ray instead")


def click_to_gaussian_idx_from_ray(ray_origin, ray_direction,
                                    client: viser.ClientHandle,
                                    renderer: GaussianRenderer,
                                    render_w: int = 960, render_h: int = 720) -> int:
    """
    Accurate click → Gaussian index using render_gaussian_idx buffer.

    viser scene = COLMAP frame, so ray_origin/ray_direction are already
    in COLMAP frame. Just project to screen UV and look up the Gaussian.
    """
    mini_cam = viser_to_minicam(client, render_w, render_h)

    ray_o = np.asarray(ray_origin, dtype=np.float32)
    ray_d = np.asarray(ray_direction, dtype=np.float32)
    ray_d = ray_d / (np.linalg.norm(ray_d) + 1e-9)

    # Pick a point along the ray (COLMAP frame = world frame for renderer)
    pt_world = ray_o + 0.1 * ray_d

    # homogeneous world point
    pt_h = torch.tensor([pt_world[0], pt_world[1], pt_world[2], 1.0],
                        dtype=torch.float32, device="cuda")

    # full_proj_transform: (4,4) maps world_h → clip_h  (column-major, so p_clip = FPT^T @ p_world)
    fpt = mini_cam.full_proj_transform   # 4×4 cuda tensor
    p_clip = fpt.T @ pt_h               # (4,)
    p_clip = p_clip.cpu().numpy()

    # Perspective divide → NDC [-1,1]
    if abs(p_clip[3]) < 1e-6:
        return -1
    ndc_x = p_clip[0] / p_clip[3]
    ndc_y = p_clip[1] / p_clip[3]

    # NDC → pixel  (NDC [-1,1] → pixel [0, W-1])
    # In standard OpenGL NDC: x=-1 is left, y=-1 is bottom
    # Viser/images: y=0 is top (image convention)
    px = int((ndc_x + 1.0) * 0.5 * render_w)
    py = int((1.0 - (ndc_y + 1.0) * 0.5) * render_h)   # flip y

    px = max(0, min(render_w - 1, px))
    py = max(0, min(render_h - 1, py))

    # Render gaussian idx map
    idx_map = renderer.render_gaussian_idx_map(mini_cam)   # H×W×N

    # Take the first (front-most) Gaussian at this pixel
    gauss_idx = int(idx_map[py, px, 0].item())
    return gauss_idx if gauss_idx >= 0 else -1


# ── Spatial segmentation ───────────────────────────────────────────────────────

def spatial_segment(xyz: np.ndarray, seed_idx: int,
                    radius: float = 0.05, min_samples: int = 10) -> np.ndarray:
    """Flood-fill from seed + DBSCAN keep largest component. Returns bool (N,)."""
    tree    = cKDTree(xyz)
    visited = np.zeros(len(xyz), dtype=bool)
    stack   = [seed_idx]
    while stack:
        i = stack.pop()
        if visited[i]:
            continue
        visited[i] = True
        for n in tree.query_ball_point(xyz[i], r=radius):
            if not visited[n]:
                stack.append(n)
    if visited.sum() < min_samples:
        return visited
    sub_xyz = xyz[visited]
    db      = DBSCAN(eps=radius * 0.7, min_samples=min_samples).fit(sub_xyz)
    labels  = db.labels_
    uniq, cnts = np.unique(labels[labels >= 0], return_counts=True)
    if len(uniq) == 0:
        return visited
    result  = np.zeros(len(xyz), dtype=bool)
    sub_idx = np.where(visited)[0]
    result[sub_idx[labels == uniq[np.argmax(cnts)]]] = True
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main(dataset: ModelParams, iteration: int, opt) -> None:

    server = viser.ViserServer()
    server.configure_theme(dark_mode=True)

    # world2base transform
    _w2b = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        dataset.model_path, "world2base.npy")
    world2base = np.load(_w2b) if os.path.exists(_w2b) else np.eye(4)
    base2colmap = np.linalg.inv(world2base)

    x_min, x_max = 0.2, 0.8
    y_min, y_max = -0.3, 0.3
    z_min, z_max = -0.02, 0.15

    # ── Load Gaussians ────────────────────────────────────────────────────────
    print("Loading Gaussian model…")

    class _Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
        gaussians.training_setup(opt)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        decoder_path = os.path.join(dataset.model_path, "feat_decoder.pth")
        assert os.path.exists(decoder_path), f"Missing feat_decoder.pth in {dataset.model_path}"
        feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=True).cuda()
        feat_decoder.load_state_dict(torch.load(decoder_path), strict=True)
        feat_decoder.eval()

        # Grab one training camera for initial viewpoint
        train_cams = scene.getTrainCameras()
        init_cam = train_cams[len(train_cams) // 2]  # middle camera
        # camera center in COLMAP world space (OpenGL convention)
        init_cam_pos = init_cam.camera_center.cpu().numpy()
        # world_view_transform is (W2V)^T; R_cw is upper-left 3x3 of W2V
        # W2V stored column-major so rows 0-2 of WVT are columns of W2V = rows of R_cw
        WVT = init_cam.world_view_transform.cpu().numpy()   # (W2V)^T
        R_cw = WVT[:3, :3].T   # rotation camera←world
        R_wc = R_cw.T          # rotation world←camera (OpenGL)
        # convert OpenGL R_wc to OpenCV for viser (flip Y,Z cols back)
        flip = np.diag([1.0, -1.0, -1.0])
        R_wc_cv = R_wc @ flip
        init_cam_wxyz = vtf.SO3.from_matrix(R_wc_cv).wxyz

    gs_renderer = GaussianRenderer(gaussians, _Pipe())

    xyz_world = gaussians.get_xyz.detach().cpu().numpy()
    xyz_base  = (world2base[:3, :3] @ xyz_world.T).T + world2base[:3, 3]

    # URDF root in COLMAP frame.
    # base2colmap has scale ~12x (COLMAP units per meter), so:
    #   position = base2colmap @ [0,0,0,1]  (robot base origin in COLMAP)
    #   rotation = pure rotation part of base2colmap
    #   urdf scale = colmap_per_meter so URDF meters match COLMAP units
    colmap_per_meter = np.linalg.norm(base2colmap[:3, :3], axis=0).mean()
    base_origin_colmap = base2colmap[:3, 3]
    R_b2c_raw = base2colmap[:3, :3]
    R_b2c_pure = R_b2c_raw / np.linalg.norm(R_b2c_raw, axis=0, keepdims=True)
    base_R_colmap = vtf.SO3.from_matrix(R_b2c_pure).wxyz

    virtual_robot = rtb.models.Panda()
    print(f"Loaded {len(xyz_world)} Gaussians.")
    print(f"Init cam pos (COLMAP): {init_cam_pos.round(3)}")

    # ── Per-client render threads ─────────────────────────────────────────────
    render_threads: Dict[int, GaussianRenderThread] = {}

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        # Set initial camera to match a training camera viewpoint
        client.camera.wxyz = tuple(init_cam_wxyz.tolist())
        client.camera.position = tuple(init_cam_pos.tolist())

        t = GaussianRenderThread(client, gs_renderer)
        render_threads[client.client_id] = t
        t.start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            if client.client_id in render_threads:
                render_threads[client.client_id].action("move")

    @server.on_client_disconnect
    def on_disconnect(client: viser.ClientHandle):
        if client.client_id in render_threads:
            render_threads[client.client_id].running = False
            render_threads.pop(client.client_id)

    def _rerender_all():
        for t in render_threads.values():
            t.action("rerender")

    # ── Segmentation / grasp state ────────────────────────────────────────────
    seg = dict(busy=False, fg_mask=None, fg_gaussians=None,
               fg_expanded=None, bg_gaussians=None,
               grasp_poses=[], grasp_scores=[])
    seg_lock = threading.Lock()

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.add_gui_folder("Click-to-Grasp"):
        seg_radius_sl = server.add_gui_slider(
            "Segment radius (m)", min=0.01, max=0.20, step=0.005, initial_value=0.05)
        click_enabled = server.add_gui_checkbox("Click-to-segment", initial_value=True)
        status_lbl    = server.add_gui_text(
            "Status", initial_value="Ready – click an object", disabled=True)
        execute_btn   = server.add_gui_button("▶ Execute Grasp on Robot")
        clear_btn     = server.add_gui_button("Clear")

    with server.add_gui_folder("Robot"):
        gui_joints: List[viser.GuiInputHandle] = []
        # Place robot base in COLMAP frame with correct scale.
        # ViserUrdf has no built-in parent frame support, so we use scale= to
        # convert URDF meters → COLMAP units, and set the root frame manually.
        server.add_frame("/panda_base",
                         wxyz=tuple(base_R_colmap.tolist()),
                         position=tuple(base_origin_colmap.tolist()),
                         show_axes=False)
        urdf = ViserUrdf(server, urdf_path=Path("./urdf/panda_newgripper.urdf"),
                         root_node_name="/panda_base",
                         scale=float(colmap_per_meter))
        for jname, (lo, hi) in urdf.get_actuated_joint_limits().items():
            lo = lo if lo is not None else -np.pi
            hi = hi if hi is not None else np.pi
            init = 1.766 if jname == "panda_joint6" else (
                0.0 if lo < 0 < hi else (lo + hi) / 2.0)
            sl = server.add_gui_slider(jname, min=lo, max=hi, step=1e-3, initial_value=init)
            sl.on_update(lambda _: urdf.update_cfg(np.array([g.value for g in gui_joints])))
            gui_joints.append(sl)
        urdf.update_cfg(np.array([g.value for g in gui_joints]))

    def _set_joints(q):
        for i, ang in enumerate(q):
            gui_joints[i].value = float(ang)

    def _set_status(msg: str):
        try:
            status_lbl.value = msg
        except Exception:
            pass

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(ray_origin, ray_direction, client: viser.ClientHandle,
                      seg_radius: float):
        _set_status("Finding Gaussian under click…")
        try:
            gauss_idx = click_to_gaussian_idx_from_ray(
                ray_origin, ray_direction, client, gs_renderer,
                render_w=960, render_h=720)

            if gauss_idx < 0:
                _set_status("Click missed – try clicking on the object surface")
                return

            _set_status(f"Seed Gaussian #{gauss_idx} – segmenting…")
            # seg_radius is in meters; xyz_world is in COLMAP units (~12x meters)
            colmap_scale = np.linalg.norm(world2base[:3, 0])  # meters/COLMAP
            radius_colmap = seg_radius / colmap_scale
            mask  = spatial_segment(xyz_world, gauss_idx, radius=radius_colmap)
            n_fg  = int(mask.sum())
            if n_fg < 20:
                _set_status(f"Segment too small ({n_fg} pts) – try larger radius")
                return

            mask_exp = edit_utils.flood_fill(xyz_world, mask, max_dist=radius_colmap * 2)

            with seg_lock:
                seg["fg_mask"]      = mask
                seg["fg_gaussians"] = edit_utils.select_gaussians(gaussians, mask)
                seg["fg_expanded"]  = edit_utils.select_gaussians(gaussians, mask_exp)
                seg["bg_gaussians"] = edit_utils.select_gaussians(gaussians, ~mask)

            gs_renderer.set_highlight(mask)
            _rerender_all()

            server.add_point_cloud(
                "pcd_fg", points=xyz_world[mask],
                colors=np.tile([255, 60, 60], (n_fg, 1)),
                point_size=0.005, position=(0, 0, 0))

            _set_status(f"Segmented {n_fg} Gaussians – generating grasps…")
            _generate_grasps(mask_exp)

        except Exception as e:
            import traceback; traceback.print_exc()
            _set_status(f"Error: {e}")
        finally:
            with seg_lock:
                seg["busy"] = False

    def _generate_grasps(mask_exp: np.ndarray):
        with seg_lock:
            g_exp = seg["fg_expanded"]
        if g_exp is None:
            return

        os.makedirs(os.path.join(dataset.model_path, "point_cloud_for_grasp"),
                    exist_ok=True)
        ply = os.path.join(dataset.model_path, "point_cloud_for_grasp/click_obj.ply")
        obj_g = edit_utils.rotate_gaussians(g_exp, world2base[:3, :3].copy())
        obj_g = edit_utils.translate_gaussians(obj_g, world2base[:3, 3])
        obj_g.save_ply(ply)

        pose_matrices, scores = grasping_utils.sample_grasps(ply, if_global=False)

        valid_poses, valid_scores = [], []
        Ry = SO3.Ry(np.pi / 2).data[0]
        for p, s in zip(pose_matrices, scores):
            gp = p.copy()
            gp[:3, :3] = p[:3, :3] @ Ry
            if np.dot(gp[:3, 0], [1, 0, 0]) < 0:
                gp[:3, :3] = gp[:3, :3] @ SO3.Rz(np.pi).data[0]
            z_vec = -gp[:3, 2]
            ang = np.arccos(np.clip(
                np.dot(z_vec / (np.linalg.norm(z_vec) + 1e-9), [0, 0, 1]), -1, 1))
            if ang > np.pi / 4:
                continue
            valid_poses.append(gp)
            valid_scores.append(s)

        with seg_lock:
            old_n = len(seg["grasp_poses"])
            seg["grasp_poses"]  = valid_poses
            seg["grasp_scores"] = valid_scores
        for i in range(old_n):
            server.add_frame(f"/grasps_{i}", wxyz=(1,0,0,0),
                             position=(0,0,0), show_axes=False, visible=False)

        if not valid_poses:
            _set_status("No valid grasps found")
            return

        scores_n = np.array(valid_scores)
        scores_n = (scores_n - scores_n.min()) / (scores_n.ptp() + 1e-9)
        dg = grasping_utils.plot_gripper_pro_max(np.zeros(3), np.eye(3), 0.08, 0.06)
        for i, (gp, sn) in enumerate(zip(valid_poses, scores_n)):
            server.add_frame(f"/grasps_{i}",
                             wxyz=vtf.SO3.from_matrix(gp[:3, :3]).wxyz,
                             position=gp[:3, 3], show_axes=False)
            server.add_mesh(f"/grasps_{i}/mesh",
                            vertices=np.asarray(dg.vertices),
                            faces=np.asarray(dg.triangles),
                            color=np.array([sn, 0.0, 1.0 - sn]))

        best = _best_reachable(valid_poses, valid_scores)
        if best >= 0:
            gp = valid_poses[best]
            roll, pitch, yaw = euler.mat2euler(gp[:3, :3])
            sol, success, *_ = virtual_robot.ik_NR(
                SE3.Trans(gp[:3, 3]) * SE3.RPY([roll, pitch, yaw]))
            if success:
                _set_joints(sol)
            np.save(os.path.join(dataset.model_path, "best_grasp_pose.npy"), gp)
            _set_status(
                f"Done – grasp #{best+1}/{len(valid_poses)}, "
                f"score={valid_scores[best]:.3f}"
                + (" | IK OK" if success else " | IK failed"))
            print(f"Best grasp → {dataset.model_path}/best_grasp_pose.npy")
        else:
            _set_status(f"{len(valid_poses)} grasps, no IK solution found")

    def _best_reachable(poses, scores) -> int:
        for i in np.argsort(scores)[::-1]:
            _, success, *_ = virtual_robot.ik_NR(
                SE3.Trans(poses[i][:3, 3]) *
                SE3.RPY(euler.mat2euler(poses[i][:3, :3])))
            if success:
                return int(i)
        return -1

    # ── Buttons ───────────────────────────────────────────────────────────────

    @execute_btn.on_click
    def _(_):
        with seg_lock:
            poses, scores = seg["grasp_poses"], seg["grasp_scores"]
        if not poses:
            print("No grasps – click an object first")
            return
        best = _best_reachable(poses, scores)
        if best < 0:
            print("No reachable grasp")
            return
        plan_utils.grasp_object(poses[best])

    @clear_btn.on_click
    def _(_):
        with seg_lock:
            old_n = len(seg["grasp_poses"])
            seg.update(fg_mask=None, fg_gaussians=None, fg_expanded=None,
                       bg_gaussians=None, grasp_poses=[], grasp_scores=[])
        gs_renderer.set_highlight(None)
        for i in range(old_n):
            server.add_frame(f"/grasps_{i}", wxyz=(1,0,0,0),
                             position=(0,0,0), show_axes=False, visible=False)
        _rerender_all()
        _set_status("Cleared – click an object")

    # ── Scene click ───────────────────────────────────────────────────────────

    @server.on_scene_click
    def on_click(event: viser.ScenePointerEvent):
        if not click_enabled.value:
            return
        with seg_lock:
            if seg["busy"]:
                _set_status("Busy – please wait")
                return
            seg["busy"] = True

        client = event.client
        threading.Thread(
            target=_run_pipeline,
            args=(event.ray_origin, event.ray_direction,
                  client, seg_radius_sl.value),
            daemon=True
        ).start()

    # ── Keep alive ────────────────────────────────────────────────────────────
    print("GraspSplats 3DGS viewer: http://localhost:8080")
    print("Click any object in the rendered scene to segment and plan a grasp.")
    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    parser = ArgumentParser(description="GraspSplats 3DGS interactive viewer")
    model = ModelParams(parser, sentinel=True)
    op    = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args  = get_combined_args(parser)
    main(model.extract(args), args.iteration, op.extract(args))
