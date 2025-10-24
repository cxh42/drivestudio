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
    import cairosvg  # optional for SVG icons
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

# 内置 SVG 图标（无需文件依赖）
BUILTIN_SVGS = {
    # 用户提供：search_hands_free_24dp_E3E3E3_FILL0_wght400_GRAD0_opsz24.svg
    "search_hands_free_24dp": (
        '<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e3e3e3">'
        '<path d="M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm-40-84v-120q-60-12-102-54t-54-102H164q12 109 89.5 185T440-164Zm80 0q109-12 186.5-89.5T796-440H676q-12 60-54 102t-102 54v120ZM164-520h116l120-120h160l120 120h116q-15-121-105-200.5T480-800q-121 0-211 79.5T164-520Z"/>'
        "</svg>"
    ),
}

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
    alpha_v: float = 0.4,
    alpha_a: float = 0.5,
    deadband: float = 0.15,
    scale_percentile: float = 90.0,
    decay: float = 0.9,
    exclusive: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据速度序列计算加速/减速强度（稳定、不易闪烁）。
    步骤：v→EMA平滑→a=diff(v)→EMA平滑→去死区→按分位数归一化→带衰减保持→(可选)互斥显示。
    返回 (acc_pos, acc_neg)，均在 [0,1]。
    """
    v = vel.astype(np.float32)
    v_s = _ema_1d(v, alpha_v)
    a = np.diff(v_s, prepend=v_s[:1])
    a_s = _ema_1d(a, alpha_a)
    # 归一化尺度
    scale = np.percentile(np.abs(a_s), float(scale_percentile)) if np.any(a_s != 0) else 1.0
    if scale <= 1e-6:
        scale = 1.0
    # 死区（相对尺度）
    th = float(deadband) * scale
    # 原始强度
    raw_pos = np.clip((a_s - th) / (scale - th + 1e-6), 0.0, 1.0)
    raw_neg = np.clip(((-a_s) - th) / (scale - th + 1e-6), 0.0, 1.0)
    # 带衰减保持，避免频繁闪烁
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
            # 互斥：谁更强亮谁，另一侧衰减更多
            if pos[i] > neg[i]:
                neg[i] *= 0.5
            elif neg[i] > pos[i]:
                pos[i] *= 0.5
    return pos, neg

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import render_images


logger = logging.getLogger("v2x_overlay")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _pick_ckpt(run_dir: str, ckpt: Optional[str]) -> str:
    if ckpt is not None:
        return ckpt
    # prefer final, else latest checkpoint by mtime
    final = os.path.join(run_dir, "checkpoint_final.pth")
    if os.path.exists(final):
        return final
    cands = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint_*.pth")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"未找到权重：{run_dir} 下不存在 checkpoint_*.pth")
    return cands[0]


def _make_vis_indices_for_first_cam(num_frames: int, num_cams: int) -> List[int]:
    """返回只渲染第一个相机的可视化索引列表（full split 下为 [0, num_cams, 2*num_cams, ...]）。"""
    return [i * num_cams + 0 for i in range(num_frames)]


def _list_yaml_frames(raw_vehicle_dir: str) -> List[str]:
    ylist = sorted(glob.glob(os.path.join(raw_vehicle_dir, "*.yaml")))
    return ylist


def _autodetect_raw_scenario(raw_root: str, expected_frames: int, vehicle_folder: str = "1") -> Optional[str]:
    """
    在 raw/<split> 下搜索场景，挑选第一个其 vehicle_folder 下 YAML 数量与期望帧数一致者。
    返回场景目录路径（不含车辆子目录）：.../raw/<split>/<scenario>
    """
    if not os.path.isdir(raw_root):
        return None
    for scenario in sorted(os.listdir(raw_root)):
        scn_dir = os.path.join(raw_root, scenario)
        veh_dir = os.path.join(scn_dir, vehicle_folder)
        if not os.path.isdir(veh_dir):
            continue
        cnt = len(_list_yaml_frames(veh_dir))
        if cnt == expected_frames and cnt > 0:
            return scn_dir
    return None


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
        # true_ego_pose: [x, y, z, roll, yaw, pitch]（度）
        pose = meta.get("true_ego_pose", meta.get("lidar_pose", None))
        if pose is None or len(pose) < 6:
            raise KeyError(f"{yp} 缺少 true_ego_pose/lidar_pose")
        yaw = float(pose[4])
        yaws_deg.append(yaw)
        # position x,y,z (meters)
        pos = [float(pose[0]), float(pose[1]), float(pose[2])]
        poss.append(pos)
        spd = meta.get("ego_speed", None)
        if spd is None:
            # 速度缺失时，填 0（后续用相对变化）
            speeds.append(0.0)
        else:
            speeds.append(float(spd))
    return (
        np.asarray(yaws_deg, dtype=np.float32),
        np.asarray(speeds, dtype=np.float32),
        np.asarray(poss, dtype=np.float32),
    )


def _compute_controls(
    yaws_deg: np.ndarray,
    speeds: np.ndarray,
    positions_xyz: Optional[np.ndarray] = None,
    yaw_gain: float = 10.0,
    wheel_fullscale_deg: float = 450.0,
    dynamic_scale: bool = False,
    wheel_ratio: float = 16.0,
    wheel_visual_gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 yaw（度）与 speed（m/s）序列估计：
    - steer_angles: 用去包裹后的 yaw 差分乘增益，近似方向盘转角（度，正左负右按右手系可调）
    - throttle: 速度正向差分归一化到 [0,1]
    - brake: 速度负向差分归一化到 [0,1]
    注：未使用真实车辆参数，主要呈现转动/加减速趋势。
    """
    # 方位角去包裹并求差分
    yaw_rad = np.deg2rad(yaws_deg)
    yaw_unwrap = np.unwrap(yaw_rad)
    dyaw_deg = np.rad2deg(np.diff(yaw_unwrap, prepend=yaw_unwrap[:1]))
    # 约定：右转为正 => 取正号；如需反向可传负的 yaw_gain
    if dynamic_scale:
        # 将序列 95 分位的 |dyaw| 映射到全幅 wheel_fullscale_deg 的 ~2/3，保留裕量
        s = np.percentile(np.abs(dyaw_deg), 95) if np.any(dyaw_deg != 0) else 1.0
        if s <= 1e-6:
            s = 1.0
        gain_eff = (wheel_fullscale_deg * (2.0 / 3.0)) / s
        steer = -dyaw_deg * gain_eff  # 右转应为顺时针，图标旋转使用 -angle，因此此处取反以匹配视觉
    else:
        # 固定比例（方向盘角：航向角 = wheel_ratio:1），如 16:1
        steer = -dyaw_deg * wheel_ratio
    # 视觉增益，再限幅为方向盘机械全幅
    steer = steer * float(max(wheel_visual_gain, 0.0))
    steer = np.clip(steer, -wheel_fullscale_deg, wheel_fullscale_deg)

    # 尝试用 ego_speed；若几乎全为 0，则用轨迹位置差分的速度替代
    sp = speeds.astype(np.float32).copy()
    if np.allclose(sp, 0.0, atol=1e-6) and positions_xyz is not None and len(positions_xyz) == len(yaws_deg):
        dp = np.linalg.norm(np.diff(positions_xyz, axis=0, prepend=positions_xyz[:1]), axis=-1)
        sp = dp  # 近似每帧位移量
    # 油门：用速度幅值归一化，确保车辆前进时有可见变化
    v_scale = np.percentile(sp, 95) if np.any(sp > 0) else 1.0
    if v_scale <= 1e-6:
        v_scale = 1.0
    throttle = np.clip(sp / v_scale, 0.0, 1.0)
    # 刹车：用速度变化的负值部分（减速）来表示
    dv = np.diff(sp, prepend=sp[:1])
    a_scale = np.percentile(np.abs(dv), 95) if np.any(dv != 0) else 1.0
    if a_scale <= 1e-6:
        a_scale = 1.0
    brake = np.clip(-dv / a_scale, 0.0, 1.0)
    return (
        steer.astype(np.float32),
        throttle.astype(np.float32),
        brake.astype(np.float32),
        sp.astype(np.float32),
    )


def _draw_steering_wheel(
    img: np.ndarray,
    angle_deg: float,
    center: Tuple[int, int],
    radius: int,
    wheel_icon_key: Optional[str] = "builtin:search_hands_free_24dp",
    rotate_icon: bool = True,
) -> None:
    """
    极简方向盘：圆环 + 单指针（只一条指针线），整体按 angle_deg 旋转，避免十字/三辐视觉。
    """
    c = center
    r = radius
    # 背景方向盘图标（可选内置 SVG），铺在圆盘区域
    size = int(r * 2)
    icon = _load_icon(wheel_icon_key, size) if wheel_icon_key is not None else None
    if icon is not None:
        icon_img = icon.copy()
        if rotate_icon and angle_deg != 0.0:
            # OpenCV 正角为逆时针；我们定义右转为正（顺时针），所以取负号
            M = cv2.getRotationMatrix2D((size // 2, size // 2), -angle_deg, 1.0)
            icon_img = cv2.warpAffine(icon_img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        _overlay_icon(img, icon_img, (c[0] + r, c[1] + r))
    else:
        # 回退：画外圈
        cv2.circle(img, c, r, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    # 不再绘制指针，仅旋转方向盘图标


def _load_icon(path: Optional[str], size: int) -> Optional[np.ndarray]:
    """加载图标为 BGRA；支持 PNG/JPG；若为 SVG 且安装了 cairosvg，则转成 PNG。
    返回 (H, W, 4) 或 None。
    """
    if path is None:
        return None
    # 支持内置 SVG：以 builtin: 前缀传入
    if isinstance(path, str) and path.startswith("builtin:"):
        key = path.split(":", 1)[1]
        svg = BUILTIN_SVGS.get(key)
        if svg is None:
            return None
        if not HAS_CAIROSVG:
            logger.warning("需要 cairosvg 才能渲染内置 SVG，已回退到占位图标。")
            return None
        try:
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
        except Exception as e:
            logger.warning(f"内置 SVG 渲染失败：{e}")
            return None
    # 文件路径
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    img = None
    if ext == ".svg":
        if not HAS_CAIROSVG:
            logger.warning("检测到 SVG 图标但未安装 cairosvg，忽略图标。")
            return None
        try:
            png_bytes = cairosvg.svg2png(url=path, output_width=size, output_height=size)
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            logger.warning(f"SVG 转 PNG 失败：{e}")
            return None
    else:
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
    """把 BGRA 图标贴到底部右侧位置（以 bottom_right 为外接点）。"""
    H, W = dst_bgr.shape[:2]
    h, w = icon_bgra.shape[:2]
    x2, y2 = bottom_right
    x1, y1 = x2 - w, y2 - h
    if x1 < 0 or y1 < 0:
        return
    roi = dst_bgr[y1:y2, x1:x2]
    # alpha blend
    alpha = icon_bgra[:, :, 3:4].astype(np.float32) / 255.0
    icon_rgb = icon_bgra[:, :, :3].astype(np.float32)
    base = roi.astype(np.float32)
    blended = icon_rgb * alpha + base * (1 - alpha)
    dst_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)


def _draw_bars(
    img: np.ndarray,
    accel_pos: float,
    accel_neg: float,
    bottom_right: Tuple[int, int],
    size: Tuple[int, int],
) -> None:
    """右下角恢复为两条竖条图：右侧为加速（绿），左侧为刹车（红）。上方居中标注文本。"""
    w, h = size
    x2, y2 = bottom_right
    gap = int(w * 0.6)
    # 左侧（刹车）矩形框
    brk_x1 = x2 - (2 * w + gap)
    thr_x1 = x2 - (w)
    y1 = y2 - h
    # 框体
    cv2.rectangle(img, (brk_x1, y1), (brk_x1 + w, y2), (200, 200, 200), 2)
    cv2.rectangle(img, (thr_x1, y1), (thr_x1 + w, y2), (200, 200, 200), 2)
    # 填充高度：按强度
    brk_fill = int(h * np.clip(accel_neg, 0.0, 1.0))
    thr_fill = int(h * np.clip(accel_pos, 0.0, 1.0))
    # 红：刹车
    cv2.rectangle(img, (brk_x1 + 2, y2 - brk_fill), (brk_x1 + w - 2, y2 - 2), (0, 0, 255), -1)
    # 绿：加速
    cv2.rectangle(img, (thr_x1 + 2, y2 - thr_fill), (thr_x1 + w - 2, y2 - 2), (0, 255, 0), -1)
    # 文本：BRK / ACC 居中到各自条上方
    def put_centered(text: str, x_left: int):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cx = x_left + w // 2
        tx = int(cx - tw / 2)
        ty = max(y1 - 8, th + 4)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    put_centered("BRK", brk_x1)
    put_centered("ACC", thr_x1)


def main():
    ap = argparse.ArgumentParser(description="渲染 v2x-real 第一视角并叠加方向盘/油门/刹车")
    ap.add_argument(
        "--run_dir",
        default="work_dirs/drivestudio/v2x-real-test-2023-04-04-14-34-53_51_1",
        help="训练输出目录（包含 config.yaml/ckpt），默认匹配 v2x-real-test-143453",
    )
    ap.add_argument("--ckpt", default=None, help="权重路径（默认优先 checkpoint_final.pth）")
    ap.add_argument("--output", default=None, help="输出 mp4 路径（默认写到 run_dir/videos/front_cam_overlay.mp4）")
    ap.add_argument("--cam_id", type=int, default=0, help="主相机编号（默认 0，前置）")
    ap.add_argument("--side_cams", type=int, nargs=3, default=[2, 3, 1], help="三个附加视角相机编号（顺序对应 labels），默认 [2,3,1]=左/后/右")
    ap.add_argument("--side_labels", type=str, nargs=3, default=["left", "back", "right"], help="三个附加视角的标签，默认 left back right")
    ap.add_argument("--cam_forward_offset", type=float, default=0.5, help="相机沿前向平移（米），默认 1.0m")
    ap.add_argument("--yaw_gain", type=float, default=10.0, help="方向盘转角增益（默认 10；当启用动态缩放时忽略）")
    ap.add_argument("--wheel_fullscale", type=float, default=450.0, help="方向盘全幅角（默认 450 度）")
    ap.add_argument("--no_dynamic_scale", action="store_true", help="关闭基于序列的动态幅度缩放")
    ap.add_argument("--steer_boost", type=float, default=3.0, help="方向盘视觉增益（在 16:1 映射基础上再放大）")
    # 加速/刹车稳定性参数
    ap.add_argument("--acc_alpha_v", type=float, default=0.4, help="速度 EMA 平滑系数 [0,1]")
    ap.add_argument("--acc_alpha_a", type=float, default=0.5, help="加速度 EMA 平滑系数 [0,1]")
    ap.add_argument("--acc_deadband", type=float, default=0.15, help="加速度死区（相对分位尺度）")
    ap.add_argument("--acc_percentile", type=float, default=90.0, help="归一化使用的分位数")
    ap.add_argument("--acc_decay", type=float, default=0.9, help="亮度衰减系数（越大保持越久）")
    # 原始数据（用于解析 yaw/speed），可留空自动匹配
    ap.add_argument("--raw_split", default="test", choices=["train1", "train4", "validate", "test"], help="raw split（默认 test）")
    ap.add_argument("--raw_root", default="data/v2x-real/raw", help="raw 数据根目录（包含各 split）")
    ap.add_argument(
        "--raw_scenario_dir",
        default="data/v2x-real/raw/test/2023-04-04-14-34-53_51_1",
        help="指定 raw/<split>/<scenario> 目录（不含车辆子目录）；默认匹配 v2x-real-test-143453",
    )
    ap.add_argument("--vehicle_folder", default="1", help="车辆文件夹名（默认 '1'，即第一辆车）")
    ap.add_argument("--fps", type=int, default=None, help="输出视频的 fps（默认取 config.render.fps）")
    # 不再使用外部图标作为油门/刹车，改为红绿灯指示
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.yaml")
    assert os.path.exists(cfg_path), f"缺少配置：{cfg_path}"
    cfg = OmegaConf.load(cfg_path)

    # dataset & trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)

    # ---- 可选：将指定相机沿其前向（Z 轴）整体平移（每帧）---- #
    if abs(args.cam_forward_offset) > 1e-6:
        try:
            cam = dataset.pixel_source.camera_data[args.cam_id]
            c2w = cam.cam_to_worlds.clone()
            forward = c2w[:, :3, 2]
            c2w[:, :3, 3] = c2w[:, :3, 3] + forward * float(args.cam_forward_offset)
            cam.cam_to_worlds = c2w
            logger.info(f"已将相机 {args.cam_id} 沿前向整体平移 {args.cam_forward_offset:.3f} m")
        except Exception as e:
            logger.warning(f"相机平移失败（忽略）：{e}")
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
    ckpt_path = _pick_ckpt(run_dir, args.ckpt)
    logger.info(f"加载权重：{ckpt_path}")
    trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
    trainer.set_eval()

    # 渲染图像（仅 RGBA/RGB，不计算指标）
    logger.info("渲染多视角序列（主视角 + 左/后/右 三视角）...")
    num_frames = dataset.num_img_timesteps
    ncam = dataset.pixel_source.num_cams
    main_indices = [i * ncam + int(args.cam_id) for i in range(num_frames)]
    side_ids = [int(x) for x in args.side_cams]
    side_indices = [[i * ncam + sid for i in range(num_frames)] for sid in side_ids]
    combined_indices = main_indices + side_indices[0] + side_indices[1] + side_indices[2]
    render_results = render_images(
        trainer=trainer,
        dataset=dataset.full_image_set,
        compute_metrics=False,
        compute_error_map=False,
        vis_indices=combined_indices,
    )
    rgbs = render_results.get("rgbs", [])
    if not rgbs:
        raise RuntimeError("render_images 返回空的 rgbs")
    T = num_frames
    expected = T * 4
    if len(rgbs) < expected:
        logger.warning(f"渲染帧数不足：rgbs={len(rgbs)}, 期望={expected}")
    # 拆分为主视角 + 三个侧视角
    main_rgbs = rgbs[0:T]
    side_rgbs = [rgbs[T:2*T], rgbs[2*T:3*T], rgbs[3*T:4*T]]

    # 解析 yaw/speed
    if args.raw_scenario_dir is None:
        # 自动匹配场景：raw_root/<split>/<scenario> 下 vehicle_folder YAML 数与 T 相等
        search_root = os.path.join(args.raw_root, args.raw_split)
        found = _autodetect_raw_scenario(search_root, expected_frames=T, vehicle_folder=args.vehicle_folder)
        if found is None:
            # 常用默认（用户提示：v2x-real-test-143453 → 2023-04-04-14-34-53_51_1）
            default_scn = os.path.join(search_root, "2023-04-04-14-34-53_51_1")
            if os.path.isdir(os.path.join(default_scn, args.vehicle_folder)):
                found = default_scn
        if found is None:
            raise FileNotFoundError(f"无法自动定位 raw 场景，请通过 --raw_scenario_dir 指定。已搜索：{search_root}")
        raw_scn_dir = found
    else:
        raw_scn_dir = args.raw_scenario_dir

    raw_vehicle_dir = os.path.join(raw_scn_dir, args.vehicle_folder)
    logger.info(f"读取控制信号：{raw_vehicle_dir}")
    yaws_deg, speeds, positions = _load_yaw_speed_from_yaml(raw_vehicle_dir)
    # 对齐帧数
    yaws_deg = yaws_deg[:T]
    speeds = speeds[:T]
    positions = positions[:T]
    steer, thr, brk, vel = _compute_controls(
        yaws_deg,
        speeds,
        positions_xyz=positions,
        yaw_gain=args.yaw_gain,
        wheel_fullscale_deg=args.wheel_fullscale,
        dynamic_scale=False,
        wheel_ratio=16.0,
        wheel_visual_gain=args.steer_boost,
    )
    # 统一加速度可视化（正=加速，负=减速），使用平滑/死区/衰减，减少闪烁
    acc_pos, acc_neg = _compute_acc_intensities(
        vel,
        alpha_v=args.acc_alpha_v,
        alpha_a=args.acc_alpha_a,
        deadband=args.acc_deadband,
        scale_percentile=args.acc_percentile,
        decay=args.acc_decay,
        exclusive=True,
    )

    # 输出路径与 fps
    out_dir = os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"front_cam_overlay_cam{args.cam_id}_with_sides.mp4")
    fps = int(args.fps or cfg.render.get("fps", 10))
    logger.info(f"写出视频：{out_path} @ {fps} fps")

    # 绘制叠加并写 mp4
    writer = imageio.get_writer(out_path, mode="I", fps=fps)
    try:
        for i in range(T):
            # 主视角
            main = (np.clip(main_rgbs[i], 0.0, 1.0) * 255).astype(np.uint8)
            main = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
            H, W = main.shape[:2]
            # 侧视角
            small_w = max(3, W // 3)
            side_imgs = []
            for j in range(3):
                s = (np.clip(side_rgbs[j][i], 0.0, 1.0) * 255).astype(np.uint8)
                s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
                h0, w0 = s.shape[:2]
                h_small = int(h0 * (small_w / float(w0)))
                s = cv2.resize(s, (small_w, h_small), interpolation=cv2.INTER_AREA)
                # 标签
                label = str(args.side_labels[j])
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                tx = max(2, (small_w - tw) // 2)
                ty = max(th + 6, th + 6)
                cv2.putText(s, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
                side_imgs.append(s)
            side_row_h = max(img.shape[0] for img in side_imgs)
            # 拼接成复合帧
            comp = np.zeros((H + side_row_h, W, 3), dtype=np.uint8)
            comp[:H, :W] = main
            # 拼接侧视角（顶对齐）
            x = 0
            for s in side_imgs:
                h_s, w_s = s.shape[:2]
                comp[H:H + h_s, x:x + w_s] = s
                x += w_s
            # 在主视角上叠加方向盘与圆点
            margin = int(0.02 * min(H, W))
            wheel_r = int(0.11 * min(H, W))
            wheel_c = (margin + wheel_r, H - margin - wheel_r)
            icon_size = int(0.09 * min(H, W))
            br_corner = (W - margin, H - margin)
            _draw_steering_wheel(comp, float(steer[i]), wheel_c, wheel_r, wheel_icon_key="builtin:search_hands_free_24dp", rotate_icon=True)
            # 两个圆点（红/绿），在主视角右下角
            def draw_dots(img, acc_p, acc_n, br, d):
                x2, y2 = br
                r = d // 2
                gap = max(6, d // 4)
                green_c = (x2 - r, y2 - r)
                red_c = (x2 - r - d - gap, y2 - r)
                for c, col in ((red_c, (180, 180, 180)), (green_c, (180, 180, 180))):
                    cv2.circle(img, c, r, col, 2, lineType=cv2.LINE_AA)
                def blend(center, color, s):
                    s = float(np.clip(s, 0.0, 1.0))
                    if s <= 1e-3:
                        return
                    overlay = img.copy()
                    cv2.circle(overlay, center, r - 3, color, -1, lineType=cv2.LINE_AA)
                    alpha = 0.15 + 0.75 * s
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                blend(green_c, (0, 255, 0), acc_pos[i])
                blend(red_c, (0, 0, 255), acc_neg[i])
                # 文本：居中到各自圆点上方
                def put_centered(text, center, color):
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    tx = int(center[0] - tw / 2)
                    ty = y2 - d - max(8, th // 2)
                    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                put_centered("BRK", red_c, (230, 230, 230))
                put_centered("ACC", green_c, (230, 230, 230))
            draw_dots(comp, float(acc_pos[i]), float(acc_neg[i]), br_corner, max(16, icon_size))

            # 输出
            frame_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
    finally:
        writer.close()
    logger.info("完成。")


if __name__ == "__main__":
    main()
