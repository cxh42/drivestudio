import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image

from utils.geometry import get_corners, project_camera_points_to_image


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_instances(inst_dir: str) -> Tuple[Dict, Dict]:
    with open(os.path.join(inst_dir, 'instances_info.json'), 'r') as f:
        instances_info = json.load(f)
    with open(os.path.join(inst_dir, 'frame_instances.json'), 'r') as f:
        frame_instances = json.load(f)
    return instances_info, frame_instances


def load_intrinsics(scene_root: str) -> Dict[int, np.ndarray]:
    intr_dir = os.path.join(scene_root, 'intrinsics')
    intrinsics: Dict[int, np.ndarray] = {}
    for name in sorted(os.listdir(intr_dir)):
        if not name.endswith('.txt'):
            continue
        cam = int(os.path.splitext(name)[0])
        vals = np.loadtxt(os.path.join(intr_dir, name))
        # file layout: fx, fy, cx, cy, k1, k2, p1, p2, k3
        fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics[cam] = K
    return intrinsics


def get_frames_and_cams(scene_root: str) -> Tuple[List[int], List[int]]:
    # Detect frames by scanning images directory
    img_dir = os.path.join(scene_root, 'images')
    frames: List[int] = []
    cams: List[int] = []
    if os.path.isdir(img_dir):
        for name in sorted(os.listdir(img_dir)):
            base = os.path.splitext(name)[0]
            if '_' not in base:
                continue
            f, c = base.split('_')
            try:
                fi, ci = int(f), int(c)
            except Exception:
                continue
            frames.append(fi)
            cams.append(ci)
    frames = sorted(sorted(set(frames)))
    cams = sorted(sorted(set(cams)))
    return frames, cams


def load_extrinsic(scene_root: str, frame: int, cam: int) -> np.ndarray:
    path = os.path.join(scene_root, 'extrinsics', f'{frame:03d}_{cam}.txt')
    return np.loadtxt(path)


def world_to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    # points_world: (N,3)
    # c2w: 4x4 -> w2c = inv(c2w)
    w2c = np.linalg.inv(c2w)
    pts = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)  # (N,4)
    pts_cam = (w2c @ pts.T).T[:, :3]
    return pts_cam


def instance_is_human(cls_name: str) -> bool:
    return cls_name.startswith('human.')


def instance_is_vehicle(cls_name: str) -> bool:
    return cls_name.startswith('vehicle.')


def main():
    ap = argparse.ArgumentParser(description='Project 3D boxes to build rough dynamic masks for V2X-Real processed scenes')
    ap.add_argument('--processed_scene', required=True, help='Path like data/v2x-real/processed/000')
    args = ap.parse_args()

    scene_root = args.processed_scene
    inst_dir = os.path.join(scene_root, 'instances')
    assert os.path.isdir(inst_dir), f'Missing instances dir: {inst_dir}. Run v2xreal_generate_instances first.'

    instances_info, frame_instances = load_instances(inst_dir)
    intrinsics = load_intrinsics(scene_root)
    frames, cams = get_frames_and_cams(scene_root)

    # Prepare output dirs
    out_all = os.path.join(scene_root, 'dynamic_masks', 'all')
    out_human = os.path.join(scene_root, 'dynamic_masks', 'human')
    out_vehicle = os.path.join(scene_root, 'dynamic_masks', 'vehicle')
    for d in (out_all, out_human, out_vehicle):
        ensure_dir(d)

    # Pre-load image sizes per cam from the first frame
    img_sizes: Dict[int, Tuple[int, int]] = {}
    for cam in cams:
        img_path = os.path.join(scene_root, 'images', f'{frames[0]:03d}_{cam}.jpg')
        if not os.path.exists(img_path):
            # try jpeg if needed
            img_path = os.path.join(scene_root, 'images', f'{frames[0]:03d}_{cam}.jpeg')
        with Image.open(img_path) as im:
            w, h = im.size
        img_sizes[cam] = (h, w)

    # Build per-frame masks
    for fi in frames:
        # Collect instances present in this frame
        key = str(fi)
        present_ids = frame_instances.get(key, [])
        # Pre-compute each instance's pose and size at this frame
        inst_cache = []
        for k in present_ids:
            k = str(k)
            info = instances_info[k]
            cls_name = info['class_name']
            fa = info['frame_annotations']
            # find this frame index within the instance
            if fi not in fa['frame_idx']:
                continue
            idx = fa['frame_idx'].index(fi)
            T_o2w = np.array(fa['obj_to_world'][idx], dtype=np.float64)
            l, w, h = fa['box_size'][idx]
            inst_cache.append((cls_name, T_o2w, (float(l), float(w), float(h))))

        # Project into each camera
        for cam in cams:
            H, W = img_sizes[cam]
            K = intrinsics[cam]
            c2w = load_extrinsic(scene_root, fi, cam)
            # masks
            mask_all = np.zeros((H, W), dtype=np.uint8)
            mask_h = np.zeros((H, W), dtype=np.uint8)
            mask_v = np.zeros((H, W), dtype=np.uint8)

            for cls_name, T_o2w, (L, Wd, Hd) in inst_cache:
                # compute corners in world
                corners_obj = get_corners(L, Wd, Hd)  # (3,8)
                corners_obj_h = np.vstack([corners_obj, np.ones((1, corners_obj.shape[1]))])  # (4,8)
                corners_w = (T_o2w @ corners_obj_h)[:3, :].T  # (8,3)
                # to camera
                corners_c = world_to_camera(corners_w, c2w)  # (8,3)
                # Define box faces using the get_corners ordering
                faces = [
                    [0, 1, 2, 3],  # top (z=+h/2)
                    [4, 5, 6, 7],  # bottom (z=-h/2)
                    [0, 1, 5, 4],  # side
                    [1, 2, 6, 5],  # side
                    [2, 3, 7, 6],  # side
                    [3, 0, 4, 7],  # side
                ]
                for face in faces:
                    pts3d = corners_c[face, :]  # (4,3)
                    # require at least some points in front
                    if (pts3d[:, 2] <= 0.1).all():
                        continue
                    pts2d, depths = project_camera_points_to_image(pts3d, K)
                    # keep only positive depth vertices
                    pos = depths > 0.1
                    if pos.sum() < 3:
                        continue
                    poly = np.round(pts2d[pos]).astype(np.int32)
                    # Clip to image range for safety
                    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
                    if len(poly) >= 3:
                        if instance_is_human(cls_name):
                            cv2.fillPoly(mask_h, [poly], 255)
                        if instance_is_vehicle(cls_name):
                            cv2.fillPoly(mask_v, [poly], 255)
            mask_all = cv2.bitwise_or(mask_h, mask_v)

            # Save
            cv2.imwrite(os.path.join(out_all, f'{fi:03d}_{cam}.png'), mask_all)
            cv2.imwrite(os.path.join(out_human, f'{fi:03d}_{cam}.png'), mask_h)
            cv2.imwrite(os.path.join(out_vehicle, f'{fi:03d}_{cam}.png'), mask_v)

    print(f'Rough dynamic masks saved under {os.path.join(scene_root, "dynamic_masks")}')


if __name__ == '__main__':
    main()
