#!/usr/bin/env python3

import os
import glob
import argparse
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

import imageio
from scipy.spatial.transform import Slerp, Rotation as R
from collections import defaultdict

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.base.pixel_source import get_rays
from datasets.base.scene_dataset import ModelType

from tools.render_v2xreal_interpfps import (
    _load_yaw_speed_from_yaml,
    _compute_acc_intensities,
    _overlay_icon,
    _load_icon,
    _draw_steering_wheel,
    _draw_acc_dots,
    interpolate_c2w_sequence,
    render_cam_sequence,
    BUILTIN_SVGS,
)


logger = logging.getLogger("v2x_interpfps_boxes")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def corners_lwh(size_lwh: np.ndarray) -> np.ndarray:
    """Return 8 corners (8,3) in object local frame given [l,w,h]."""
    l, w, h = size_lwh.astype(np.float32)
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=np.float32)
    return corners


def slerp_se3(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    """Interpolate SE(3): translation linear, rotation slerp, fraction a in [0,1]."""
    t0, t1 = T0[:3, 3], T1[:3, 3]
    R0, R1 = T0[:3, :3], T1[:3, :3]
    key = R.from_matrix(np.stack([R0, R1], axis=0))
    s = Slerp([0.0, 1.0], key)
    Rm = s(a).as_matrix()
    tm = (1.0 - a) * t0 + a * t1
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rm.astype(np.float32)
    out[:3, 3] = tm.astype(np.float32)
    return out


def project_points(world_pts: np.ndarray, c2w: np.ndarray, K: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Nx3 world points to image pixels; return (pixels Nx2, cam_pts Nx3, valid_mask Nx bool)."""
    w2c = np.linalg.inv(c2w).astype(np.float32)
    pts_h = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    x = cam[:, 0] / np.maximum(z, 1e-6)
    y = cam[:, 1] / np.maximum(z, 1e-6)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    px = np.stack([u, v], axis=1)
    valid &= (u >= -10) & (u < W + 10) & (v >= -10) & (v < H + 10)
    return px, cam, valid


def draw_box(img: np.ndarray, cam_pts: np.ndarray, K: np.ndarray, scale_xy: Tuple[float, float], img_size: Tuple[int, int], color=(0, 255, 255), thickness=1) -> None:
    """Draw 3D box edges with near-plane clipping and optional scaling.
    cam_pts: (8,3) in camera coords; K: 3x3; scale_xy: (sx, sy) to resize to target canvas; img_size: (H, W).
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    eps = 1e-3
    sx, sy = scale_xy
    H, W = img_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    def proj(pt):
        x, y, z = pt
        u = fx * (x / max(z, 1e-6)) + cx
        v = fy * (y / max(z, 1e-6)) + cy
        return np.array([u * sx, v * sy], dtype=np.float32)
    # Cohen–Sutherland style 2D clipping helpers
    LEFT, RIGHT, BOTTOM, TOP = 1, 2, 4, 8
    def outcode(p):
        code = 0
        if p[0] < 0: code |= LEFT
        elif p[0] >= W: code |= RIGHT
        if p[1] < 0: code |= TOP
        elif p[1] >= H: code |= BOTTOM
        return code
    def clip_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        while True:
            c1, c2 = outcode((x1, y1)), outcode((x2, y2))
            if not (c1 | c2):
                return (x1, y1), (x2, y2)
            if c1 & c2:
                return None
            c = c1 or c2
            if c & TOP:
                x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1 + 1e-8)
                y = 0
            elif c & BOTTOM:
                x = x1 + (x2 - x1) * ((H - 1) - y1) / (y2 - y1 + 1e-8)
                y = H - 1
            elif c & RIGHT:
                y = y1 + (y2 - y1) * ((W - 1) - x1) / (x2 - x1 + 1e-8)
                x = W - 1
            else:  # LEFT
                y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1 + 1e-8)
                x = 0
            if c == c1:
                x1, y1 = x, y
            else:
                x2, y2 = x, y

    for i, j in edges:
        p0 = cam_pts[i].copy()
        p1 = cam_pts[j].copy()
        z0, z1 = p0[2], p1[2]
        if z0 <= eps and z1 <= eps:
            continue
        # clip against z=eps
        if z0 <= eps:
            t = (eps - z0) / (z1 - z0 + 1e-8)
            p0 = p0 + t * (p1 - p0)
            p0[2] = eps
        if z1 <= eps:
            t = (eps - z1) / (z0 - z1 + 1e-8)
            p1 = p1 + t * (p0 - p1)
            p1[2] = eps
        u0 = proj(p0)
        u1 = proj(p1)
        pt1 = (float(u0[0]), float(u0[1]))
        pt2 = (float(u1[0]), float(u1[1]))
        # clip to image rect to避免极端情况下的巨大线段
        clipped = clip_line(pt1, pt2)
        if clipped is None:
            continue
        (x1, y1), (x2, y2) = clipped
        cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness, lineType=cv2.LINE_AA)

def find_ego_instance(poses: np.ndarray, cam_c2w: np.ndarray, visible: np.ndarray, dist_thresh: float = 3.0) -> Optional[int]:
    """Return instance index that is closest (median distance) to main camera across frames; if below threshold.
    poses: (T,N,4,4); cam_c2w: (T,4,4); visible: (T,N)
    """
    T, N = poses.shape[0], poses.shape[1]
    cam_xyz = cam_c2w[:, :3, 3]  # (T,3)
    med_dists = []
    for k in range(N):
        obj_xyz = poses[:, k, :3, 3]
        # 仅用可见帧计算距离，若没有可见帧则用全帧
        mask = visible[:, k]
        if mask.sum() == 0:
            diff = obj_xyz - cam_xyz
        else:
            diff = obj_xyz[mask] - cam_xyz[mask]
        d = np.linalg.norm(diff, axis=-1)
        med_dists.append(np.median(d) if d.size > 0 else 1e9)
    ego_idx = int(np.argmin(med_dists)) if N > 0 else None
    if N == 0:
        return None
    return ego_idx if med_dists[ego_idx] < dist_thresh else None


def collect_instance_boxes(dataset: DrivingDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps = dataset.pixel_source
    poses = ps.instances_pose.cpu().numpy().astype(np.float32)  # (T, N, 4,4)
    sizes = ps.instances_size.cpu().numpy().astype(np.float32)  # (N,3)
    visible = ps.per_frame_instance_mask.cpu().numpy().astype(bool)  # (T,N)
    types = ps.instances_model_types.cpu().numpy()  # (N,)
    return poses, sizes, visible, types


def main():
    ap = argparse.ArgumentParser(description="时间插帧渲染 + 3D车辆边界框（主视+左/后/右）")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target_fps", type=int, default=60)
    ap.add_argument("--orig_fps", type=float, default=None)
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--side_cams", type=int, nargs=3, default=[2, 3, 1])
    ap.add_argument("--side_labels", type=str, nargs=3, default=["left", "back", "right"])
    ap.add_argument("--cam_forward_offset", type=float, default=1.0)
    ap.add_argument("--steer_boost", type=float, default=3.0)
    ap.add_argument("--wheel_fullscale", type=float, default=450.0)
    ap.add_argument("--raw_scenario_dir", default="data/v2x-real/raw/test/2023-04-04-14-34-53_51_1")
    # smoothing params
    ap.add_argument("--acc_alpha_v", type=float, default=0.2)
    ap.add_argument("--acc_alpha_a", type=float, default=0.3)
    ap.add_argument("--acc_deadband", type=float, default=0.25)
    ap.add_argument("--acc_percentile", type=float, default=85.0)
    ap.add_argument("--acc_decay", type=float, default=0.96)
    ap.add_argument("--acc_final_alpha", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=None)
    # box culling params
    ap.add_argument("--box_max_dist", type=float, default=40.0, help="超过此距离(米)不显示边界框")
    ap.add_argument("--box_min_diag", type=float, default=12.0, help="投影后对角线像素小于此阈值则不显示")
    ap.add_argument("--ego_near_thresh", type=float, default=2.5, help="自车近距离剔除阈值(米)，避免偶发显示")
    ap.add_argument("--box_min_frames", type=int, default=3, help="至少连续通过剔除阈值的帧数才开始绘制，抑制闪烁")
    ap.add_argument("--box_smooth_alpha", type=float, default=0.5, help="边界框顶点在相机坐标的EMA平滑系数(0..1)，越大越平滑")
    ap.add_argument("--box_max_speed", type=float, default=45.0, help="原始帧间中心速度阈值(米/秒)以上不绘制(可过滤闪现噪声)")
    ap.add_argument("--temporal_samples", type=int, default=3, help="每帧时间超采样数（>1 将做时间平均，近似运动模糊）")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.yaml")
    assert os.path.exists(cfg_path), f"缺少配置：{cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer: BasicTrainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    ckpt = os.path.join(run_dir, "checkpoint_final.pth")
    if not os.path.exists(ckpt):
        cands = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pth")))
        assert cands, f"未找到权重：{run_dir}"
        ckpt = cands[-1]
    logger.info(f"加载权重：{ckpt}")
    trainer.resume_from_checkpoint(ckpt_path=ckpt, load_only_model=True)
    trainer.set_eval()

    # 原始帧数与fps
    T = dataset.num_img_timesteps
    orig_fps = float(args.orig_fps or cfg.render.get("fps", 10))
    tgt_fps = int(args.target_fps)
    fps_ratio = float(tgt_fps) / float(orig_fps)
    T_new = int(round((T - 1) * fps_ratio)) + 1
    logger.info(f"原帧数={T}, 原fps={orig_fps} => 目标fps={tgt_fps}, 新帧数={T_new}")

    # 相机
    main_cam = dataset.pixel_source.camera_data[args.cam_id]
    H, W = main_cam.HEIGHT, main_cam.WIDTH
    intr_main = main_cam.intrinsics[0].to(device)
    c2w_main = main_cam.cam_to_worlds.clone()
    if abs(args.cam_forward_offset) > 1e-6:
        fwd = c2w_main[:, :3, 2]
        c2w_main[:, :3, 3] = c2w_main[:, :3, 3] + fwd * float(args.cam_forward_offset)

    side_ids = [int(x) for x in args.side_cams]
    side_cams = [dataset.pixel_source.camera_data[sid] for sid in side_ids]
    intr_sides = [cam.intrinsics[0].to(device) for cam in side_cams]
    HW_sides = [(cam.HEIGHT, cam.WIDTH) for cam in side_cams]
    c2w_sides = [cam.cam_to_worlds.clone() for cam in side_cams]

    # 插值轨迹
    traj_main = interpolate_c2w_sequence(c2w_main.to(device), T_new)
    traj_sides = [interpolate_c2w_sequence(c2w.to(device), T_new) for c2w in c2w_sides]

    # 渲染
    main_imgs = render_cam_sequence(trainer, intr_main, H, W, traj_main, device)
    side_imgs = [
        render_cam_sequence(trainer, intr_sides[j], HW_sides[j][0], HW_sides[j][1], traj_sides[j], device)
        for j in range(3)
    ]
    # 时间超采样近似：对序列做时间窗口平均（盒子叠加保持清晰，仍用中心时刻）
    if args.temporal_samples and args.temporal_samples > 1:
        K = int(args.temporal_samples)
        def blur_seq(imgs: List[np.ndarray]) -> List[np.ndarray]:
            Tn = len(imgs)
            acc = [imgs[i].astype(np.float32) for i in range(Tn)]
            out = []
            half = K // 2
            for i in range(Tn):
                s = max(0, i - half)
                e = min(Tn, i - half + K)
                if e - s < K:
                    left_pad = max(0, half - i)
                    right_pad = max(0, (i - half + K) - Tn)
                    idxs = list(range(s, e))
                    if left_pad:
                        idxs = [s] * left_pad + idxs
                    if right_pad:
                        idxs = idxs + [e - 1] * right_pad
                else:
                    idxs = list(range(s, e))
                avg = sum(acc[j] for j in idxs) / float(K)
                out.append(np.clip(avg, 0, 255).astype(np.uint8))
            return out
        main_imgs = blur_seq(main_imgs)
        side_imgs = [blur_seq(seq) for seq in side_imgs]

    # HUD 驱动（若 YAML 不存在则回退到主摄轨迹推算）
    t_tgt = np.linspace(0, 1, T_new)
    try:
        raw_vehicle_dir = os.path.join(args.raw_scenario_dir, "1")
        yaws_deg, speeds, positions = _load_yaw_speed_from_yaml(raw_vehicle_dir)
        yaw_rad = np.deg2rad(yaws_deg)
        yaw_unwrap = np.unwrap(yaw_rad)
        t_src = np.linspace(0, 1, len(yaw_unwrap))
        yaw_interp = np.interp(t_tgt, t_src, yaw_unwrap)
        dyaw_deg = np.rad2deg(np.diff(yaw_interp, prepend=yaw_interp[:1]))
        v_interp = np.interp(t_tgt, np.linspace(0, 1, len(speeds)), speeds)
    except Exception:
        raise RuntimeError("缺少 V2X-Real YAML，无法计算 HUD。请提供 --raw_scenario_dir")
    steer = -dyaw_deg * fps_ratio * 16.0 * float(max(args.steer_boost, 0.0))
    steer = np.clip(steer, -args.wheel_fullscale, args.wheel_fullscale)
    acc_pos, acc_neg = _compute_acc_intensities(
        v_interp,
        alpha_v=args.acc_alpha_v,
        alpha_a=args.acc_alpha_a,
        deadband=args.acc_deadband,
        scale_percentile=args.acc_percentile,
        decay=args.acc_decay,
        exclusive=True,
        final_alpha=args.acc_final_alpha,
    )

    # 实例盒数据
    poses, sizes, visible, types = collect_instance_boxes(dataset)
    veh_mask = (types == int(ModelType.RigidNodes))
    sizes = sizes[veh_mask]
    visible = visible[:, veh_mask]
    poses = poses[:, veh_mask]
    N_inst = sizes.shape[0]
    box_local = [corners_lwh(sizes[i]) for i in range(N_inst)]
    # 识别并排除自车（与主摄最近的实例）
    cam_c2w_orig = dataset.pixel_source.camera_data[args.cam_id].cam_to_worlds.cpu().numpy()
    ego_idx = find_ego_instance(poses, cam_c2w_orig, visible, dist_thresh=3.0)
    if ego_idx is not None:
        logger.info(f"排除自车实例 idx={ego_idx}")
    # 预计算原始速度(米/秒)用于过滤异常快速轨迹
    speeds_mps = np.zeros((poses.shape[0]-1, N_inst), dtype=np.float32) if poses.shape[0] > 1 else np.zeros((0, N_inst), dtype=np.float32)
    if poses.shape[0] > 1:
        for t in range(poses.shape[0]-1):
            c0 = poses[t, :, :3, 3]
            c1 = poses[t+1, :, :3, 3]
            speeds_mps[t] = np.linalg.norm(c1 - c0, axis=-1) * orig_fps

    # 平滑与计数缓存
    prev_campts_main: dict[int, np.ndarray] = {}
    prev_campts_sides: List[dict] = [dict(), dict(), dict()]
    count_main = defaultdict(int)
    count_sides = [defaultdict(int), defaultdict(int), defaultdict(int)]

    # 输出视频
    out_dir = os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"front_cam_interp_fps_{tgt_fps}_with_sides_boxes.mp4")
    writer = imageio.get_writer(out_path, mode="I", fps=int(args.fps or tgt_fps))

    try:
        for i in range(T_new):
            main = cv2.cvtColor(main_imgs[i], cv2.COLOR_RGB2BGR)
            Hm, Wm = main.shape[:2]
            # 侧视行
            small_w = max(3, Wm // 3)
            side_row = []
            for j in range(3):
                s = side_imgs[j][i]
                s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
                h0, w0 = s.shape[:2]
                hs = int(h0 * (small_w / float(w0)))
                s = cv2.resize(s, (small_w, hs), interpolation=cv2.INTER_AREA)
                # 标签
                label = str(args.side_labels[j])
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                tx = max(2, (small_w - tw) // 2)
                ty = max(th + 6, th + 6)
                cv2.putText(s, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
                side_row.append(s)
            side_h = max(img.shape[0] for img in side_row)
            comp = np.zeros((Hm + side_h, Wm, 3), dtype=np.uint8)
            comp[:Hm, :Wm] = main
            x = 0
            for s in side_row:
                hs, ws = s.shape[:2]
                comp[Hm: Hm + hs, x: x + ws] = s
                x += ws

            # HUD on main
            margin = int(0.02 * min(Hm, Wm))
            wheel_r = int(0.11 * min(Hm, Wm))
            wheel_c = (margin + wheel_r, Hm - margin - wheel_r)
            icon_size = int(0.09 * min(Hm, Wm))
            br_corner = (Wm - margin, Hm - margin)
            _draw_steering_wheel(comp, float(steer[i]), wheel_c, wheel_r, wheel_icon_key="builtin:search_hands_free_24dp", rotate_icon=True)
            _draw_acc_dots(comp, float(acc_pos[i]), float(acc_neg[i]), br_corner, max(16, icon_size))

            # 3D 盒子：主视角
            # 计算当前归一化时间和对应原序列索引/权重
            t = i / max(1, T_new - 1)
            idx_float = t * (T - 1)
            i0 = int(np.floor(idx_float))
            i1 = min(T - 1, i0 + 1)
            a = float(idx_float - i0)
            # 相机矩阵
            K_main = intr_main.detach().cpu().numpy()
            c2w_m = traj_main[i].detach().cpu().numpy()
            # 绘制每个车辆
            for k in range(N_inst):
                if ego_idx is not None and k == ego_idx:
                    continue
                if not (visible[i0, k] or visible[i1, k]):
                    continue
                # 原始速度过滤（异常快速跳变的轨迹不画）
                if i0 < speeds_mps.shape[0] and speeds_mps[i0, k] > args.box_max_speed:
                    count_main[k] = 0
                    continue
                T0 = poses[i0, k]
                T1 = poses[i1, k]
                o2w = slerp_se3(T0, T1, a)
                corners = box_local[k]  # (8,3)
                corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
                world = (o2w @ corners_h.T).T[:, :3]
                # 距离剔除与自车近距离剔除（基于主摄）
                center = o2w[:3, 3]
                dist_main = float(np.linalg.norm(center - c2w_m[:3, 3]))
                if dist_main > args.box_max_dist or dist_main < args.ego_near_thresh:
                    count_main[k] = 0
                    continue
                pts2d, campts, valid = project_points(world, c2w_m, K_main, Hm, Wm)
                # 最小屏幕尺寸剔除
                umin, vmin = np.min(pts2d, axis=0)
                umax, vmax = np.max(pts2d, axis=0)
                diag = float(np.hypot(umax - umin, vmax - vmin))
                if diag < args.box_min_diag:
                    count_main[k] = 0
                    continue
                # 连续帧计数 + 平滑
                count_main[k] += 1
                if count_main[k] < args.box_min_frames:
                    prev_campts_main[k] = campts
                    continue
                if k in prev_campts_main:
                    a_s = float(np.clip(args.box_smooth_alpha, 0.0, 1.0))
                    campts = a_s * campts + (1.0 - a_s) * prev_campts_main[k]
                prev_campts_main[k] = campts
                draw_box(comp[:Hm, :Wm], campts, K_main, (1.0, 1.0), (Hm, Wm), color=(0, 255, 255), thickness=1)

            # 侧视角盒子（投影在各自小图上）
            x_offset = 0
            for j in range(3):
                sH, sW = side_row[j].shape[:2]
                K_side = intr_sides[j].detach().cpu().numpy()
                c2w_s = traj_sides[j][i].detach().cpu().numpy()
                canvas = comp[Hm: Hm + sH, x_offset: x_offset + sW]
                for k in range(N_inst):
                    if ego_idx is not None and k == ego_idx:
                        continue
                    if not (visible[i0, k] or visible[i1, k]):
                        continue
                    if i0 < speeds_mps.shape[0] and speeds_mps[i0, k] > args.box_max_speed:
                        count_sides[j][k] = 0
                        continue
                    T0 = poses[i0, k]
                    T1 = poses[i1, k]
                    o2w = slerp_se3(T0, T1, a)
                    world = (o2w @ np.concatenate([box_local[k], np.ones((8, 1), dtype=np.float32)], axis=1).T).T[:, :3]
                    # 先按原分辨率投影，再按缩放到小图的比例重映射；距离与尺寸剔除同样适用
                    # 以当前侧摄为参考做距离剔除
                    dist_side = float(np.linalg.norm(o2w[:3, 3] - c2w_s[:3, 3]))
                    if dist_side > args.box_max_dist or dist_side < args.ego_near_thresh:
                        count_sides[j][k] = 0
                        continue
                    pts2d_full, campts, valid = project_points(world, c2w_s, K_side, HW_sides[j][0], HW_sides[j][1])
                    umin, vmin = np.min(pts2d_full, axis=0)
                    umax, vmax = np.max(pts2d_full, axis=0)
                    diag = float(np.hypot(umax - umin, vmax - vmin))
                    if diag < args.box_min_diag:
                        count_sides[j][k] = 0
                        continue
                    # 缩放到 sW 宽（等比）
                    scale = sW / float(HW_sides[j][1])
                    # 连续帧计数 + 平滑
                    count_sides[j][k] += 1
                    if count_sides[j][k] < args.box_min_frames:
                        prev_campts_sides[j][k] = campts
                        continue
                    if k in prev_campts_sides[j]:
                        a_s = float(np.clip(args.box_smooth_alpha, 0.0, 1.0))
                        campts = a_s * campts + (1.0 - a_s) * prev_campts_sides[j][k]
                    prev_campts_sides[j][k] = campts
                    draw_box(canvas, campts, K_side, (scale, scale), (sH, sW), color=(0, 255, 255), thickness=1)
                x_offset += sW

            writer.append_data(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    logger.info(f"已保存：{out_path}")


if __name__ == "__main__":
    main()
