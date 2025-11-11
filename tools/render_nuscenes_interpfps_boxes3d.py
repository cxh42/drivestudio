#!/usr/bin/env python3

import os, glob, argparse, logging
from typing import Dict, Tuple
import numpy as np
import torch
import cv2, imageio
from omegaconf import OmegaConf
from scipy.spatial.transform import Slerp, Rotation as R

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from datasets.base.pixel_source import get_rays
from datasets.base.scene_dataset import ModelType
from tools.render_v2xreal_interpfps import _draw_steering_wheel

logger = logging.getLogger("nuscenes_boxes3d")
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


def render_cam_sequence(trainer: BasicTrainer, cam_intr: torch.Tensor, H: int, W: int, traj: torch.Tensor, device) -> list:
    x, y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    x_flat, y_flat = x.flatten(), y.flatten()
    imgs = []
    for i in range(traj.shape[0]):
        c2w = traj[i]
        origins, viewdirs, direction_norm = get_rays(x_flat, y_flat, c2w, cam_intr)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        direction_norm = direction_norm.reshape(H, W, 1)
        cam_infos = {"camera_to_world": c2w, "intrinsics": cam_intr, "height": torch.tensor([H], device=device), "width": torch.tensor([W], device=device)}
        image_infos = {"origins": origins, "viewdirs": viewdirs, "direction_norm": direction_norm, "img_idx": torch.full((H, W), i, device=device), "frame_idx": torch.full((H, W), i, device=device), "normed_time": torch.full((H, W), i / max(1, traj.shape[0] - 1), device=device), "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1)}
        with torch.no_grad():
            rgb = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=True)["rgb"].detach().cpu().numpy().clip(1e-6, 1-1e-6)
            imgs.append((rgb * 255).astype(np.uint8))
    return imgs


def corners_lwh(size_lwh: np.ndarray) -> np.ndarray:
    l, w, h = size_lwh.astype(np.float32)
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    return np.array([
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
    ], dtype=np.float32)


def slerp_se3(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
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


def draw_box(img: np.ndarray, cam_pts: np.ndarray, K: np.ndarray, scale_xy: Tuple[float, float], img_size: Tuple[int, int], color=(0, 255, 255), thickness=1) -> None:
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
                x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1 + 1e-8); y = 0
            elif c & BOTTOM:
                x = x1 + (x2 - x1) * ((H - 1) - y1) / (y2 - y1 + 1e-8); y = H - 1
            elif c & RIGHT:
                y = y1 + (y2 - y1) * ((W - 1) - x1) / (x2 - x1 + 1e-8); x = W - 1
            else:  # LEFT
                y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1 + 1e-8); x = 0
            if c == c1: x1, y1 = x, y
            else: x2, y2 = x, y
    for i, j in edges:
        p0 = cam_pts[i].copy(); p1 = cam_pts[j].copy()
        z0, z1 = p0[2], p1[2]
        if z0 <= eps and z1 <= eps: continue
        if z0 <= eps:
            t = (eps - z0) / (z1 - z0 + 1e-8)
            p0 = p0 + t * (p1 - p0); p0[2] = eps
        if z1 <= eps:
            t = (eps - z1) / (z0 - z1 + 1e-8)
            p1 = p1 + t * (p0 - p1); p1[2] = eps
        u0 = proj(p0); u1 = proj(p1)
        pt1 = (float(u0[0]), float(u0[1])); pt2 = (float(u1[0]), float(u1[1]))
        clipped = clip_line(pt1, pt2)
        if clipped is None: continue
        (x1, y1), (x2, y2) = clipped
        cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness, lineType=cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="NuScenes 插帧渲染 + 3D框 + HUD + 四视图")
    ap.add_argument("--run_dir", required=True); ap.add_argument("--target_fps", type=int, default=60)
    ap.add_argument("--orig_fps", type=float, default=None); ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--side_cams", type=int, nargs=3, default=[1,5,2]); ap.add_argument("--side_labels", type=str, nargs=3, default=["left","back","right"])
    # steering smoothing（更平滑）
    ap.add_argument("--steer_boost", type=float, default=1.0); ap.add_argument("--wheel_fullscale", type=float, default=450.0)
    ap.add_argument("--steer_alpha", type=float, default=0.15); ap.add_argument("--steer_final_alpha", type=float, default=0.25)
    ap.add_argument("--steer_idle_speed", type=float, default=0.3); ap.add_argument("--steer_rate_limit_deg", type=float, default=1.2)
    ap.add_argument("--steer_min_dyaw_deg", type=float, default=0.05); ap.add_argument("--steer_sign", type=float, default=-1.0)
    # acc/brk 更平滑
    ap.add_argument("--acc_alpha_v", type=float, default=0.12); ap.add_argument("--acc_alpha_a", type=float, default=0.25)
    ap.add_argument("--acc_deadband", type=float, default=0.35); ap.add_argument("--acc_percentile", type=float, default=85.0)
    ap.add_argument("--acc_decay", type=float, default=0.90); ap.add_argument("--acc_final_alpha", type=float, default=0.30)
    # 3D框参数（抑制闪烁）
    ap.add_argument("--box_max_dist", type=float, default=45.0)
    ap.add_argument("--box_min_diag", type=float, default=10.0)
    ap.add_argument("--ego_near_thresh", type=float, default=2.5)
    ap.add_argument("--box_min_frames", type=int, default=3)
    ap.add_argument("--box_smooth_alpha", type=float, default=0.6)
    ap.add_argument("--box_max_speed", type=float, default=45.0)
    # 刹车门控参数（只在接近停车时点亮）
    ap.add_argument("--brk_only_stop", action="store_true", default=True)
    ap.add_argument("--brk_stop_speed", type=float, default=0.6)
    ap.add_argument("--brk_decel_min", type=float, default=0.3)
    ap.add_argument("--brk_lookahead_s", type=float, default=1.2)
    ap.add_argument("--brk_hold_s", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=None); ap.add_argument("--temporal_samples", type=int, default=3)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer: BasicTrainer = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model, num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set), test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2,3), device=device)
    ckpt = os.path.join(args.run_dir, "checkpoint_final.pth");
    if not os.path.exists(ckpt):
        cands = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint_*.pth"))); assert cands, "no ckpt"; ckpt = cands[-1]
    trainer.resume_from_checkpoint(ckpt_path=ckpt, load_only_model=True); trainer.set_eval()

    # 帧率与目标帧数
    T = dataset.num_img_timesteps; orig_fps = float(args.orig_fps or cfg.render.get("fps", 10)); tgt_fps = int(args.target_fps)
    fps_ratio = float(tgt_fps) / float(orig_fps); T_new = int(round((T - 1) * fps_ratio)) + 1

    # 相机
    main_cam = dataset.pixel_source.camera_data[args.cam_id]; H,W = main_cam.HEIGHT, main_cam.WIDTH; intr_main = main_cam.intrinsics[0].to(device)
    c2w_main = main_cam.cam_to_worlds.clone()
    side_ids=[int(x) for x in args.side_cams]
    side_cams=[dataset.pixel_source.camera_data[sid] for sid in side_ids]
    intr_sides=[cam.intrinsics[0].to(device) for cam in side_cams]
    HW_sides=[(cam.HEIGHT, cam.WIDTH) for cam in side_cams]
    c2w_sides=[cam.cam_to_worlds.clone() for cam in side_cams]

    traj_main = interpolate_c2w_sequence(c2w_main.to(device), T_new); traj_sides=[interpolate_c2w_sequence(c2w.to(device), T_new) for c2w in c2w_sides]
    main_imgs = render_cam_sequence(trainer, intr_main, H, W, traj_main, device)
    side_imgs = [render_cam_sequence(trainer, intr_sides[j], HW_sides[j][0], HW_sides[j][1], traj_sides[j], device) for j in range(3)]
    if args.temporal_samples and args.temporal_samples>1:
        K=int(args.temporal_samples)
        def blur_seq(imgs):
            Tn=len(imgs); acc=[imgs[i].astype(np.float32) for i in range(Tn)]; out=[]; half=K//2
            for i in range(Tn):
                s=max(0,i-half); e=min(Tn,i-half+K); idxs=list(range(s,e))
                if e-s<K:
                    if i<half: idxs=[s]*(half-i)+idxs
                    if i-half+K>Tn: idxs=idxs+[e-1]*(i-half+K-Tn)
                avg=sum(acc[j] for j in idxs)/float(K); out.append(np.clip(avg,0,255).astype(np.uint8))
            return out
        main_imgs=blur_seq(main_imgs); side_imgs=[blur_seq(seq) for seq in side_imgs]

    # 轨迹航向/速度 → HUD（更平滑参数）
    pos_i = traj_main[:, :3, 3].detach().cpu().numpy(); dp_i = np.diff(pos_i, axis=0, prepend=pos_i[:1])
    v_interp = np.linalg.norm(dp_i, axis=-1) * float(tgt_fps)
    yaw_i = []
    for i in range(T_new):
        Rm = traj_main[i].detach().cpu().numpy()[:3,:3]
        yaw_i.append(np.arctan2(Rm[2,2], Rm[0,2]))
    yaw_i = np.unwrap(np.array(yaw_i, dtype=np.float32))
    dyaw_deg = np.rad2deg(np.diff(yaw_i, prepend=yaw_i[:1]))
    mask_keep = (np.abs(v_interp) >= float(args.steer_idle_speed)) | (np.abs(dyaw_deg) >= float(args.steer_min_dyaw_deg))
    dyaw_deg = dyaw_deg * mask_keep.astype(np.float32)
    if args.steer_rate_limit_deg and args.steer_rate_limit_deg>0:
        lim=float(args.steer_rate_limit_deg); dyaw_deg=np.clip(dyaw_deg,-lim,lim)
    dyaw_deg=_ema_1d(dyaw_deg, args.steer_alpha)
    sign = -1.0 if float(args.steer_sign) < 0 else 1.0
    steer = sign * dyaw_deg * float(fps_ratio) * 16.0 * float(max(args.steer_boost, 0.0))
    steer = np.clip(steer, -args.wheel_fullscale, args.wheel_fullscale)
    steer = _ema_1d(steer, args.steer_final_alpha)
    acc_pos, acc_neg = _compute_acc_intensities(v_interp, args.acc_alpha_v, args.acc_alpha_a, args.acc_deadband, args.acc_percentile, args.acc_decay, True, args.acc_final_alpha)
    # Brake gating
    L = max(1, int(round(float(args.brk_lookahead_s) * tgt_fps)))
    v_min_future = np.array([np.min(v_interp[i:min(T_new, i+L)]) for i in range(T_new)])
    stop_mask = v_min_future < float(args.brk_stop_speed)
    dv_inst = np.diff(v_interp, prepend=v_interp[:1])
    trend_mask = dv_inst < -float(args.brk_decel_min)
    trigger = stop_mask & trend_mask
    hold_frames = max(1, int(round(float(args.brk_hold_s) * tgt_fps)))
    gate = np.zeros(T_new, dtype=np.float32)
    on=False; cool=0
    for i in range(T_new):
        if trigger[i]: on=True; cool=hold_frames
        if on:
            gate[i]=1.0; cool-=1
            if cool<=0 and not trigger[i]: on=False
    acc_neg_disp = acc_neg * gate

    # 3D框准备：实例信息
    ps = dataset.pixel_source
    poses = ps.instances_pose.cpu().numpy().astype(np.float32)      # (T, N, 4,4)
    sizes = ps.instances_size.cpu().numpy().astype(np.float32)      # (N,3)
    visible = ps.per_frame_instance_mask.cpu().numpy().astype(bool) # (T,N)
    types = ps.instances_model_types.cpu().numpy()                  # (N,)
    N = sizes.shape[0]
    # 识别自车实例（若有）：选距离主摄最近的一个
    cam_xyz = c2w_main[:, :3, 3].cpu().numpy()
    med_d = []
    for k in range(N):
        obj_xyz = poses[:, k, :3, 3]
        d = np.linalg.norm(obj_xyz - cam_xyz, axis=-1)
        med_d.append(float(np.median(d)))
    ego_idx = int(np.argmin(med_d)) if len(med_d)>0 else None
    if ego_idx is not None and med_d[ego_idx] > args.ego_near_thresh:
        ego_idx = None

    # 为每个(视图, 实例)保存EMA平滑的相机坐标顶点 + 连续帧计数
    ema_cam_corners: Dict[Tuple[int,int], np.ndarray] = {}
    pass_count: Dict[Tuple[int,int], int] = {}

    # 输出
    out_dir=os.path.join(args.run_dir,'videos'); os.makedirs(out_dir,exist_ok=True)
    out_path=os.path.join(out_dir,f"front_cam_interp_fps_{tgt_fps}_with_sides_boxes.mp4")
    writer=imageio.get_writer(out_path, mode='I', fps=int(args.fps or tgt_fps))
    try:
        for i in range(T_new):
            main=cv2.cvtColor(main_imgs[i], cv2.COLOR_RGB2BGR); Hm,Wm=main.shape[:2]
            small_w=max(3,Wm//3); side_row=[]
            for j in range(3):
                s=side_imgs[j][i]; s=cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
                h0,w0=s.shape[:2]; hs=int(h0*(small_w/float(w0))); s=cv2.resize(s,(small_w,hs),interpolation=cv2.INTER_AREA)
                label=str(args.side_labels[j]); (tw,th),_=cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
                tx=max(2,(small_w-tw)//2); ty=max(th+6, th+6); cv2.putText(s,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.55,(230,230,230),1,cv2.LINE_AA)
                side_row.append(s)
            side_h=max(img.shape[0] for img in side_row); comp=np.zeros((Hm+side_h,Wm,3),dtype=np.uint8)
            comp[:Hm,:Wm]=main; x=0
            for s in side_row:
                hs,ws=s.shape[:2]; comp[Hm: Hm+hs, x:x+ws]=s; x+=ws

            # HUD：方向盘+圆点（更平滑参数）
            margin=int(0.02*min(Hm,Wm)); wheel_r=int(0.11*min(Hm,Wm)); wheel_c=(margin+wheel_r, Hm-margin-wheel_r)
            _draw_steering_wheel(comp, float(steer[i]), wheel_c, wheel_r, wheel_icon_key="builtin:search_hands_free_24dp", rotate_icon=True)
            icon_size=int(0.09*min(Hm,Wm)); r=icon_size//2; gap=max(6, icon_size//4)
            green_c=(Wm-r-margin, Hm-r-margin); red_c=(Wm-r-icon_size-gap-margin, Hm-r-margin)
            for c in (green_c, red_c): cv2.circle(comp, c, r, (180,180,180), 2, lineType=cv2.LINE_AA)
            def blend(center, color, s):
                s = float(np.clip(s, 0.0, 1.0))
                if s <= 1e-3:
                    return
                overlay = comp.copy()
                cv2.circle(overlay, center, r - 3, color, -1, lineType=cv2.LINE_AA)
                alpha = 0.15 + 0.75 * s
                cv2.addWeighted(overlay, alpha, comp, 1 - alpha, 0, comp)
            blend(green_c,(0,255,0), acc_pos[i]); blend(red_c,(0,0,255), acc_neg_disp[i])
            (tw,th),_=cv2.getTextSize("ACC", cv2.FONT_HERSHEY_SIMPLEX,0.55,1); cv2.putText(comp,"ACC",(green_c[0]-tw//2, Hm-icon_size-max(8,th//2)), cv2.FONT_HERSHEY_SIMPLEX,0.55,(230,230,230),1,cv2.LINE_AA)
            (tw,th),_=cv2.getTextSize("BRK", cv2.FONT_HERSHEY_SIMPLEX,0.55,1); cv2.putText(comp,"BRK",(red_c[0]-tw//2, Hm-icon_size-max(8,th//2)), cv2.FONT_HERSHEY_SIMPLEX,0.55,(230,230,230),1,cv2.LINE_AA)

            # 3D框：主视图 + 侧视图
            view_list = [(0, traj_main[i].detach().cpu().numpy(), intr_main.detach().cpu().numpy(), (1.0, 1.0), (Hm, Wm), (0, 0, Hm, Wm))]
            x=0
            for j in range(3):
                hs, ws = side_row[j].shape[:2]
                scale_xy = (ws/float(HW_sides[j][1]), hs/float(HW_sides[j][0]))
                view_list.append((j+1, traj_sides[j][i].detach().cpu().numpy(), intr_sides[j].detach().cpu().numpy(), scale_xy, (hs, ws), (Hm, x, hs, ws)))
                x += ws

            t_src = (i / fps_ratio)
            t0 = int(np.clip(np.floor(t_src), 0, T-1)); t1 = int(np.clip(t0+1, 0, T-1)); a = float(np.clip(t_src - t0, 0.0, 1.0))
            for (view_id, c2w, K, scale_xy, img_size, region) in view_list:
                Hc, Wc = img_size
                y_off, x_off, _, _ = region
                w2c = np.linalg.inv(c2w).astype(np.float32)
                for k in range(N):
                    if types[k] != ModelType.RigidNodes:  # 车辆为主
                        continue
                    if ego_idx is not None and k == ego_idx:
                        continue
                    vis_and = bool(visible[t0, k] and visible[t1, k])
                    if not vis_and:
                        pass_count[(view_id,k)] = 0
                        continue
                    T0 = poses[t0, k]; T1 = poses[t1, k]
                    Tm = slerp_se3(T0, T1, a)
                    center = Tm[:3, 3]
                    cam_center = (w2c @ np.append(center, 1.0))[:3]
                    if np.linalg.norm(cam_center) > args.box_max_dist:
                        pass_count[(view_id,k)] = 0
                        continue
                    # 速度估计（用原始帧中心差）
                    v = np.linalg.norm(poses[t1,k,:3,3] - poses[t0,k,:3,3]) * orig_fps
                    if v > args.box_max_speed:
                        pass_count[(view_id,k)] = 0
                        continue
                    # 角点 in world
                    c_local = corners_lwh(sizes[k])
                    c_world = (Tm @ np.concatenate([c_local, np.ones((8,1),dtype=np.float32)], axis=1).T).T[:, :3]
                    # 到相机坐标
                    cam_pts = (w2c @ np.concatenate([c_world, np.ones((8,1),dtype=np.float32)], axis=1).T).T[:, :3]
                    # 投影对角线判断
                    fx, fy, cx0, cy0 = K[0,0], K[1,1], K[0,2], K[1,2]
                    def proj(p):
                        z = max(p[2], 1e-6); return np.array([fx*(p[0]/z)+cx0, fy*(p[1]/z)+cy0], dtype=np.float32)
                    u = np.stack([proj(cam_pts[idx]) for idx in [0,2,4,6]], axis=0)  # 四个角大概
                    diag = float(np.linalg.norm((u.max(0)-u.min(0)) * np.array([scale_xy[0], scale_xy[1]])))
                    if diag < args.box_min_diag:
                        pass_count[(view_id,k)] = 0
                        continue
                    # 连续帧与EMA平滑
                    key=(view_id,k)
                    if key not in ema_cam_corners:
                        ema_cam_corners[key]=cam_pts.copy()
                        pass_count[key]=1
                    else:
                        ema_cam_corners[key]= args.box_smooth_alpha*cam_pts + (1-args.box_smooth_alpha)*ema_cam_corners[key]
                        pass_count[key]+=1
                    if pass_count[key] < int(args.box_min_frames):
                        continue
                    # 绘制
                    color=(0,255,255)
                    if view_id==0:
                        draw_box(comp, ema_cam_corners[key], K, (1.0,1.0), (Hm, Wm), color=color, thickness=1)
                    else:
                        draw_box(comp[y_off:y_off+Hc, x_off:x_off+Wc], ema_cam_corners[key], K, scale_xy, (Hc, Wc), color=color, thickness=1)

            writer.append_data(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    finally:
        writer.close(); logger.info(f"已保存：{out_path}")


if __name__ == '__main__':
    main()
