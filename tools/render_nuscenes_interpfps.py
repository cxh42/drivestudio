#!/usr/bin/env python3

import os
import glob
import argparse
import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tools.render_v2xreal_interpfps import _draw_steering_wheel
import imageio
from scipy.spatial.transform import Slerp, Rotation as R

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.base.pixel_source import get_rays

logger = logging.getLogger("nuscenes_interpfps")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    alpha_v: float = 0.12,
    alpha_a: float = 0.25,
    deadband: float = 0.35,
    scale_percentile: float = 85.0,
    decay: float = 0.90,
    exclusive: bool = True,
    final_alpha: float = 0.30,
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
    pos = _ema_1d(pos, final_alpha)
    neg = _ema_1d(neg, final_alpha)
    return pos, neg


def interpolate_c2w_sequence(c2w_seq: torch.Tensor, target_frames: int) -> torch.Tensor:
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
    ap = argparse.ArgumentParser(description="NuScenes 插帧渲染（主视+左/后/右），HUD 自动驱动")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target_fps", type=int, default=60)
    ap.add_argument("--orig_fps", type=float, default=None)
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--side_cams", type=int, nargs=3, default=[1, 5, 2])
    ap.add_argument("--side_labels", type=str, nargs=3, default=["left", "back", "right"])
    ap.add_argument("--steer_boost", type=float, default=1.0, help="方向盘放大系数（相对16:1基准的额外倍数）")
    ap.add_argument("--wheel_fullscale", type=float, default=450.0)
    ap.add_argument("--steer_alpha", type=float, default=0.15, help="方向变化EMA平滑系数[0..1]（对dyaw做平滑）")
    ap.add_argument("--steer_idle_speed", type=float, default=0.3, help="低于该速度(m/s)判定静止，抑制乱转")
    ap.add_argument("--steer_rate_limit_deg", type=float, default=1.2, help="每帧最大航向变化(度)，进一步抑制尖峰")
    ap.add_argument("--steer_min_dyaw_deg", type=float, default=0.05, help="即使低速也保留的最小dyaw阈值(度/帧)")
    ap.add_argument("--steer_sign", type=float, default=-1.0, help="方向盘方向因子（+1或-1；默认-1与v2x-real相反）")
    ap.add_argument("--steer_final_alpha", type=float, default=0.25, help="对最终方向盘角再做一次EMA，进一步平滑")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--temporal_samples", type=int, default=3)
    # 指示灯平滑参数（更稳更少闪烁）
    ap.add_argument("--acc_alpha_v", type=float, default=0.12)
    ap.add_argument("--acc_alpha_a", type=float, default=0.25)
    ap.add_argument("--acc_deadband", type=float, default=0.35)
    ap.add_argument("--acc_percentile", type=float, default=85.0)
    ap.add_argument("--acc_decay", type=float, default=0.90)
    ap.add_argument("--acc_final_alpha", type=float, default=0.30)
    # Brake gating（只在“确实要停车”时点亮）
    ap.add_argument("--brk_only_stop", action="store_true", default=True)
    ap.add_argument("--brk_stop_speed", type=float, default=0.6, help="低于该速度(米/秒)视为接近停车")
    ap.add_argument("--brk_decel_min", type=float, default=0.3, help="瞬时减速度阈值(米/秒^2)，小于-阈值才认为在刹车")
    ap.add_argument("--brk_lookahead_s", type=float, default=1.2, help="前瞻秒数，若未来窗口内将低于停止速度则允许点亮")
    ap.add_argument("--brk_hold_s", type=float, default=0.5, help="刹车灯最小保持秒数，避免闪烁")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
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
    trainer.resume_from_checkpoint(ckpt_path=ckpt, load_only_model=True)
    trainer.set_eval()

    # 帧率与轨迹
    T = dataset.num_img_timesteps
    orig_fps = float(args.orig_fps or cfg.render.get("fps", 10))
    tgt_fps = int(args.target_fps)
    fps_ratio = float(tgt_fps) / float(orig_fps)
    T_new = int(round((T - 1) * fps_ratio)) + 1

    main_cam = dataset.pixel_source.camera_data[args.cam_id]
    H, W = main_cam.HEIGHT, main_cam.WIDTH
    intr_main = main_cam.intrinsics[0].to(device)
    c2w_main = main_cam.cam_to_worlds.clone()
    side_ids = [int(x) for x in args.side_cams]
    side_cams = [dataset.pixel_source.camera_data[sid] for sid in side_ids]
    intr_sides = [cam.intrinsics[0].to(device) for cam in side_cams]
    HW_sides = [(cam.HEIGHT, cam.WIDTH) for cam in side_cams]
    c2w_sides = [cam.cam_to_worlds.clone() for cam in side_cams]

    traj_main = interpolate_c2w_sequence(c2w_main.to(device), T_new)
    traj_sides = [interpolate_c2w_sequence(c2w.to(device), T_new) for c2w in c2w_sides]

    main_imgs = render_cam_sequence(trainer, intr_main, H, W, traj_main, device)
    side_imgs = [
        render_cam_sequence(trainer, intr_sides[j], HW_sides[j][0], HW_sides[j][1], traj_sides[j], device)
        for j in range(3)
    ]
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
                idxs = list(range(s, e))
                if e - s < K:
                    if i < half:
                        idxs = [s] * (half - i) + idxs
                    if i - half + K > Tn:
                        idxs = idxs + [e - 1] * (i - half + K - Tn)
                avg = sum(acc[j] for j in idxs) / float(K)
                out.append(np.clip(avg, 0, 255).astype(np.uint8))
            return out
        main_imgs = blur_seq(main_imgs)
        side_imgs = [blur_seq(seq) for seq in side_imgs]

    # HUD（基于插值后的轨迹计算航向与速度；对dyaw进行解包、限幅与EMA）
    pos_i = traj_main[:, :3, 3].detach().cpu().numpy()
    # 目标帧间隔是 1/tgt_fps，速度用目标fps计算更一致
    dp_i = np.diff(pos_i, axis=0, prepend=pos_i[:1])
    v_interp = np.linalg.norm(dp_i, axis=-1) * float(tgt_fps)
    # 航向从旋转矩阵直接计算并解包
    yaw_i = []
    for i in range(T_new):
        Rm = traj_main[i].detach().cpu().numpy()[:3, :3]
        # 使用前向轴（第3列）在XZ平面的投影计算航向（Y 轴为竖直），更符合数据坐标系
        yaw_i.append(np.arctan2(Rm[2, 2], Rm[0, 2]))
    yaw_i = np.unwrap(np.array(yaw_i, dtype=np.float32))
    # 直接对dyaw做平滑更不易“幅度被吃掉”
    dyaw_deg = np.rad2deg(np.diff(yaw_i, prepend=yaw_i[:1]))
    # 低速抑制但保留明显转向：速度低且角变化也很小才抑制
    mask_keep = (np.abs(v_interp) >= float(args.steer_idle_speed)) | (np.abs(dyaw_deg) >= float(args.steer_min_dyaw_deg))
    dyaw_deg = dyaw_deg * mask_keep.astype(np.float32)
    # 每帧航向变化限幅以抑制尖峰
    if args.steer_rate_limit_deg is not None and args.steer_rate_limit_deg > 0:
        lim = float(args.steer_rate_limit_deg)
        dyaw_deg = np.clip(dyaw_deg, -lim, lim)
    # EMA 平滑
    dyaw_deg = _ema_1d(dyaw_deg, args.steer_alpha)
    # 方向盘角度：16:1，并可选左右反转
    sign = -1.0 if float(args.steer_sign) < 0 else 1.0
    # 放大到与原fps一致的感受：逐帧航向差随帧率增大而变小，这里乘以 fps_ratio 补偿
    steer = sign * dyaw_deg * float(fps_ratio) * 16.0 * float(max(args.steer_boost, 0.0))
    steer = np.clip(steer, -args.wheel_fullscale, args.wheel_fullscale)
    # 对最终方向盘角再做一次EMA，进一步平滑
    steer = _ema_1d(steer, args.steer_final_alpha)
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
    # 刹车只在“确实要停车”时点亮：速度前瞻 + 减速度门限 + 保持
    if args.brk_only_stop:
        L = max(1, int(round(float(args.brk_lookahead_s) * tgt_fps)))
        v_min_future = np.array([np.min(v_interp[i:min(T_new, i + L)]) for i in range(T_new)])
        stop_mask = v_min_future < float(args.brk_stop_speed)
        dv_inst = np.diff(v_interp, prepend=v_interp[:1])
        trend_mask = dv_inst < -float(args.brk_decel_min)
        trigger = stop_mask & trend_mask
        hold_frames = max(1, int(round(float(args.brk_hold_s) * tgt_fps)))
        gate = np.zeros(T_new, dtype=np.float32)
        on = False
        cool = 0
        for i in range(T_new):
            if trigger[i]:
                on = True
                cool = hold_frames
            if on:
                gate[i] = 1.0
                cool -= 1
                if cool <= 0 and not trigger[i]:
                    on = False
        acc_neg_disp = acc_neg * gate
    else:
        acc_neg_disp = acc_neg

    out_dir = os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"front_cam_interp_fps_{tgt_fps}_with_sides.mp4")
    writer = imageio.get_writer(out_path, mode='I', fps=int(args.fps or tgt_fps))
    try:
        for i in range(T_new):
            main = cv2.cvtColor(main_imgs[i], cv2.COLOR_RGB2BGR)
            Hm, Wm = main.shape[:2]
            small_w = max(3, Wm // 3)
            side_row = []
            for j in range(3):
                s = side_imgs[j][i]
                s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
                h0, w0 = s.shape[:2]
                hs = int(h0 * (small_w / float(w0)))
                s = cv2.resize(s, (small_w, hs), interpolation=cv2.INTER_AREA)
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

            # HUD 绘制（回到原先样式：旋转方向盘图标）
            margin = int(0.02 * min(Hm, Wm))
            wheel_r = int(0.11 * min(Hm, Wm))
            wheel_c = (margin + wheel_r, Hm - margin - wheel_r)
            icon_size = int(0.09 * min(Hm, Wm))
            br_corner = (Wm - margin, Hm - margin)
            _draw_steering_wheel(comp, float(steer[i]), wheel_c, wheel_r, wheel_icon_key="builtin:search_hands_free_24dp", rotate_icon=True)
            # ACC/BRK 圆点
            r = icon_size // 2
            gap = max(6, icon_size // 4)
            green_c = (Wm - r - margin, Hm - r - margin)
            red_c = (Wm - r - icon_size - gap - margin, Hm - r - margin)
            for c in (green_c, red_c):
                cv2.circle(comp, c, r, (180, 180, 180), 2, lineType=cv2.LINE_AA)
            def blend(center, color, s):
                s = float(np.clip(s, 0.0, 1.0))
                if s <= 1e-3: return
                overlay = comp.copy()
                cv2.circle(overlay, center, r - 3, color, -1, lineType=cv2.LINE_AA)
                alpha = 0.15 + 0.75 * s
                cv2.addWeighted(overlay, alpha, comp, 1 - alpha, 0, comp)
            blend(green_c, (0,255,0), acc_pos[i])
            blend(red_c, (0,0,255), acc_neg_disp[i])
            # labels
            (tw, th), _ = cv2.getTextSize("ACC", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(comp, "ACC", (green_c[0] - tw//2, Hm - icon_size - max(8, th//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize("BRK", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(comp, "BRK", (red_c[0] - tw//2, Hm - icon_size - max(8, th//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)

            writer.append_data(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    logger.info(f"已保存：{out_path}")


if __name__ == "__main__":
    main()
