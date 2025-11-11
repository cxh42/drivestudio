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
try:
    import cairosvg
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.base.pixel_source import get_rays

from scipy.spatial.transform import Slerp, Rotation as R


logger = logging.getLogger("v2x_interpfps")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# 内置方向盘 SVG
BUILTIN_SVGS = {
    "search_hands_free_24dp": (
        '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3">'
        '<path d="M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm-40-84v-120q-60-12-102-54t-54-102H164q12 109 89.5 185T440-164Zm80 0q109-12 186.5-89.5T796-440H676q-12 60-54 102t-102 54v120ZM164-520h116l120-120h160l120 120h116q-15-121-105-200.5T480-800q-121 0-211 79.5T164-520Z"/>'
        "</svg>"
    ),
}


def _list_yaml_frames(raw_vehicle_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(raw_vehicle_dir, "*.yaml")))


def _load_yaw_speed_from_yaml(raw_vehicle_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_YAML:
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml")
    files = _list_yaml_frames(raw_vehicle_dir)
    if not files:
        raise FileNotFoundError(f"未找到 YAML: {raw_vehicle_dir}")
    yaws_deg: List[float] = []
    speeds: List[float] = []
    poss: List[List[float]] = []
    for yp in files:
        with open(yp, "r") as f:
            meta = yaml.safe_load(f)
        pose = meta.get("true_ego_pose", meta.get("lidar_pose", None))
        if pose is None or len(pose) < 6:
            raise KeyError(f"{yp} 缺少 true_ego_pose/lidar_pose")
        yaws_deg.append(float(pose[4]))
        poss.append([float(pose[0]), float(pose[1]), float(pose[2])])
        spd = meta.get("ego_speed", 0.0)
        speeds.append(float(spd) if spd is not None else 0.0)
    return (
        np.asarray(yaws_deg, dtype=np.float32),
        np.asarray(speeds, dtype=np.float32),
        np.asarray(poss, dtype=np.float32),
    )


def _ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0:
        return y
    y[0] = x[0]
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def _compute_acc_intensities(
    vel: np.ndarray,
    alpha_v: float = 0.2,
    alpha_a: float = 0.3,
    deadband: float = 0.25,
    scale_percentile: float = 85.0,
    decay: float = 0.96,
    exclusive: bool = True,
    final_alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    v = vel.astype(np.float32)
    v_s = _ema_1d(v, alpha_v)
    a = np.diff(v_s, prepend=v_s[:1])
    a_s = _ema_1d(a, alpha_a)
    scale = np.percentile(np.abs(a_s), float(scale_percentile)) if np.any(a_s != 0) else 1.0
    if scale <= 1e-6:
        scale = 1.0
    th = float(deadband) * scale
    raw_pos = np.clip((a_s - th) / (scale - th + 1e-6), 0.0, 1.0)
    raw_neg = np.clip(((-a_s) - th) / (scale - th + 1e-6), 0.0, 1.0)
    pos = np.zeros_like(raw_pos)
    neg = np.zeros_like(raw_neg)
    dec = float(np.clip(decay, 0.0, 1.0))
    for i in range(len(raw_pos)):
        if i == 0:
            pos[i] = raw_pos[i]
            neg[i] = raw_neg[i]
        else:
            pos[i] = max(raw_pos[i], pos[i - 1] * dec)
            neg[i] = max(raw_neg[i], neg[i - 1] * dec)
        if exclusive:
            if pos[i] > neg[i]:
                neg[i] *= 0.5
            elif neg[i] > pos[i]:
                pos[i] *= 0.5
    # 最后一层低通（再次 EMA），进一步抑制残余抖动
    pos = _ema_1d(pos, final_alpha)
    neg = _ema_1d(neg, final_alpha)
    return pos, neg


def _load_icon(path: Optional[str], size: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    if isinstance(path, str) and path.startswith("builtin:"):
        key = path.split(":", 1)[1]
        svg = BUILTIN_SVGS.get(key)
        if svg is None or not HAS_CAIROSVG:
            return None
        png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size)
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _overlay_icon(dst_bgr: np.ndarray, icon_bgra: np.ndarray, bottom_right: Tuple[int, int]) -> None:
    H, W = dst_bgr.shape[:2]
    h, w = icon_bgra.shape[:2]
    x2, y2 = bottom_right
    x1, y1 = x2 - w, y2 - h
    if x1 < 0 or y1 < 0:
        return
    roi = dst_bgr[y1:y2, x1:x2]
    alpha = icon_bgra[:, :, 3:4].astype(np.float32) / 255.0
    icon_rgb = icon_bgra[:, :, :3].astype(np.float32)
    base = roi.astype(np.float32)
    blended = icon_rgb * alpha + base * (1 - alpha)
    dst_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)


def _draw_steering_wheel(
    img: np.ndarray,
    angle_deg: float,
    center: Tuple[int, int],
    radius: int,
    wheel_icon_key: Optional[str] = "builtin:search_hands_free_24dp",
    rotate_icon: bool = True,
) -> None:
    c = center
    r = radius
    size = int(r * 2)
    icon = _load_icon(wheel_icon_key, size) if wheel_icon_key is not None else None
    if icon is not None:
        icon_img = icon.copy()
        if rotate_icon and angle_deg != 0.0:
            M = cv2.getRotationMatrix2D((size // 2, size // 2), -angle_deg, 1.0)
            icon_img = cv2.warpAffine(icon_img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        _overlay_icon(img, icon_img, (c[0] + r, c[1] + r))
    else:
        cv2.circle(img, c, r, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def _draw_acc_dots(img: np.ndarray, acc_p: float, acc_n: float, br_corner: Tuple[int, int], diameter: int) -> None:
    x2, y2 = br_corner
    r = diameter // 2
    gap = max(6, diameter // 4)
    green_c = (x2 - r, y2 - r)
    red_c = (x2 - r - diameter - gap, y2 - r)
    for c in (green_c, red_c):
        cv2.circle(img, c, r, (180, 180, 180), 2, lineType=cv2.LINE_AA)
    def blend(center, color, s):
        s = float(np.clip(s, 0.0, 1.0))
        if s <= 1e-3:
            return
        overlay = img.copy()
        cv2.circle(overlay, center, r - 3, color, -1, lineType=cv2.LINE_AA)
        alpha = 0.15 + 0.75 * s
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    blend(green_c, (0, 255, 0), acc_p)
    blend(red_c, (0, 0, 255), acc_n)
    # labels centered
    def put_centered(text, center, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = int(center[0] - tw / 2)
        ty = y2 - diameter - max(8, th // 2)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    put_centered("BRK", red_c, (230, 230, 230))
    put_centered("ACC", green_c, (230, 230, 230))


def interpolate_c2w_sequence(c2w_seq: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Slerp rotations + linear translation across the whole sequence."""
    device = c2w_seq.device
    pose_np = c2w_seq.detach().cpu().numpy()
    T = pose_np.shape[0]
    times = np.linspace(0.0, 1.0, T)
    tgt = np.linspace(0.0, 1.0, target_frames)
    trans = pose_np[:, :3, 3]
    interp_t = np.stack([np.interp(tgt, times, trans[:, i]) for i in range(3)], axis=-1)
    rots = R.from_matrix(pose_np[:, :3, :3])
    slerp = Slerp(times, rots)
    interp_R = slerp(tgt).as_matrix()
    out = np.eye(4)[None].repeat(target_frames, axis=0)
    out[:, :3, :3] = interp_R
    out[:, :3, 3] = interp_t
    return torch.tensor(out, dtype=torch.float32, device=device)


def render_cam_sequence(trainer: BasicTrainer, cam_intr: torch.Tensor, H: int, W: int, traj: torch.Tensor, device) -> List[np.ndarray]:
    """Render a sequence given intrinsics and c2w trajectory."""
    # Precompute grid
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    x_flat, y_flat = x.flatten(), y.flatten()
    imgs = []
    for i in range(traj.shape[0]):
        c2w = traj[i]
        origins, viewdirs, direction_norm = get_rays(x_flat, y_flat, c2w, cam_intr)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        direction_norm = direction_norm.reshape(H, W, 1)
        cam_infos = {
            "camera_to_world": c2w,
            "intrinsics": cam_intr,
            "height": torch.tensor([H], dtype=torch.long, device=device),
            "width": torch.tensor([W], dtype=torch.long, device=device),
        }
        image_infos = {
            "origins": origins,
            "viewdirs": viewdirs,
            "direction_norm": direction_norm,
            "img_idx": torch.full((H, W), i, dtype=torch.long, device=device),
            "frame_idx": torch.full((H, W), i, dtype=torch.long, device=device),
            "normed_time": torch.full((H, W), i / max(1, traj.shape[0] - 1), dtype=torch.float32, device=device),
            "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1),
        }
        with torch.no_grad():
            outputs = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=True)
            rgb = outputs["rgb"].detach().cpu().numpy().clip(1e-6, 1-1e-6)
            imgs.append((rgb * 255).astype(np.uint8))
    return imgs


def main():
    ap = argparse.ArgumentParser(description="渲染阶段时间插帧（主视角+左/后/右），保持时长并提升帧率")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target_fps", type=int, default=60)
    ap.add_argument("--orig_fps", type=float, default=None, help="原始fps，默认取config.render.fps")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--side_cams", type=int, nargs=3, default=[2, 3, 1])
    ap.add_argument("--side_labels", type=str, nargs=3, default=["left", "back", "right"])
    ap.add_argument("--cam_forward_offset", type=float, default=1.0, help="仅主摄前移（m）")
    ap.add_argument("--steer_boost", type=float, default=3.0)
    ap.add_argument("--wheel_fullscale", type=float, default=450.0)
    ap.add_argument("--raw_scenario_dir", default="data/v2x-real/raw/test/2023-04-04-14-34-53_51_1")
    ap.add_argument("--fps", type=int, default=None, help="输出封装FPS，默认=target_fps")
    ap.add_argument("--temporal_samples", type=int, default=3, help="每帧时间超采样数（>1 将做时间平均，近似运动模糊）")
    # 加速灯稳定性
    ap.add_argument("--acc_alpha_v", type=float, default=0.2)
    ap.add_argument("--acc_alpha_a", type=float, default=0.3)
    ap.add_argument("--acc_deadband", type=float, default=0.25)
    ap.add_argument("--acc_percentile", type=float, default=85.0)
    ap.add_argument("--acc_decay", type=float, default=0.96)
    ap.add_argument("--acc_final_alpha", type=float, default=0.25)
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
        # fallback latest checkpoint_*.pth
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

    # 主摄与侧摄相机内参/分辨率
    main_cam = dataset.pixel_source.camera_data[args.cam_id]
    H, W = main_cam.HEIGHT, main_cam.WIDTH
    intr_main = main_cam.intrinsics[0].to(device)

    # 获取原始c2w并可选前移（主摄）
    c2w_main_orig = main_cam.cam_to_worlds.clone()
    c2w_main = c2w_main_orig.clone()
    if abs(args.cam_forward_offset) > 1e-6:
        fwd = c2w_main[:, :3, 2]
        c2w_main[:, :3, 3] = c2w_main[:, :3, 3] + fwd * float(args.cam_forward_offset)

    # 侧摄参数
    side_ids = [int(x) for x in args.side_cams]
    side_cams = [dataset.pixel_source.camera_data[sid] for sid in side_ids]
    intr_sides = [cam.intrinsics[0].to(device) for cam in side_cams]
    HW_sides = [(cam.HEIGHT, cam.WIDTH) for cam in side_cams]
    c2w_sides = [cam.cam_to_worlds.clone() for cam in side_cams]

    # 计算插值轨迹
    traj_main = interpolate_c2w_sequence(c2w_main.to(device), T_new)
    traj_sides = [interpolate_c2w_sequence(c2w.to(device), T_new) for c2w in c2w_sides]

    # 渲染
    main_imgs = render_cam_sequence(trainer, intr_main, H, W, traj_main, device)
    side_imgs = [
        render_cam_sequence(trainer, intr_sides[j], HW_sides[j][0], HW_sides[j][1], traj_sides[j], device)
        for j in range(3)
    ]
    # 简易时间超采样（运动模糊近似）：对渲染序列做时间窗口平均
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
                    # pad on both sides by clamping indices
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

    # HUD 驱动：插值 yaw/speed 到新时间轴
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
    # 注意：更高帧率下逐帧航向差会按 dt 变小，为保持感知幅度一致，乘以 fps_ratio
    # 方向盘角：右转顺时针（负号匹配OpenCV旋转）
    steer = -dyaw_deg * fps_ratio * 16.0 * float(max(args.steer_boost, 0.0))
    steer = np.clip(steer, -args.wheel_fullscale, args.wheel_fullscale)
    # 速度稳定指示
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

    # 输出视频
    out_dir = os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"front_cam_interp_fps_{tgt_fps}_with_sides.mp4")
    fps_out = int(args.fps or tgt_fps)
    writer = imageio.get_writer(out_path, mode="I", fps=fps_out)
    try:
        for i in range(T_new):
            main = main_imgs[i]
            Hm, Wm = main.shape[:2]
            main_bgr = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
            # 侧视缩放
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
            comp[:Hm, :Wm] = main_bgr
            x = 0
            for s in side_row:
                hs, ws = s.shape[:2]
                comp[Hm: Hm + hs, x: x + ws] = s
                x += ws
            # HUD：方向盘与圆点
            margin = int(0.02 * min(Hm, Wm))
            wheel_r = int(0.11 * min(Hm, Wm))
            wheel_c = (margin + wheel_r, Hm - margin - wheel_r)
            icon_size = int(0.09 * min(Hm, Wm))
            br_corner = (Wm - margin, Hm - margin)
            _draw_steering_wheel(comp, float(steer[i]), wheel_c, wheel_r, wheel_icon_key="builtin:search_hands_free_24dp", rotate_icon=True)
            _draw_acc_dots(comp, float(acc_pos[i]), float(acc_neg[i]), br_corner, max(16, icon_size))
            writer.append_data(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    logger.info(f"已保存：{out_path}")


if __name__ == "__main__":
    main()
