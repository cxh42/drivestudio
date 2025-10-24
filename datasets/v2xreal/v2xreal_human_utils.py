from typing import List
import os
import cv2
import json
import logging
import numpy as np
from tqdm import tqdm

from utils.geometry import (
    get_corners,
    project_camera_points_to_image,
)

logger = logging.getLogger()

# For V2X-Real processed format we follow the NuScenes-style layout
# Coordinate conversion OpenCV->Dataset: identity for our processed extrinsics
OPENCV2DATASET = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)

# Human classes stored by the converter (see datasets/tools/v2xreal_to_processed_multicam.py)
SMPLNODE_CLASSES = ["human.pedestrian.adult"]

# Default camera list for typical v2x-real scenes (12 cams). If a scene has a
# different count, you can pass an explicit list to project_human_boxes via caller.
AVAILABLE_CAM_LIST = list(range(12))
CAMERA_LIST = AVAILABLE_CAM_LIST


def project_human_boxes(
    scene_dir: str,
    camera_list: List[int],
    save_temp: bool = True,
    verbose: bool = False,
    narrow_width_ratio: float = 0.2,
    fps: int = 12,
):
    """Project 3D human boxes to 2D images and dump per-cam json.

    Expects processed layout:
      - images/{frame}_{cam}.jpg
      - intrinsics/{cam}.txt (fx,fy,cx,cy,k1..k3)
      - extrinsics/{frame}_{cam}.txt (4x4 cam->world)
      - instances/{instances_info.json, frame_instances.json}
    """
    images_dir = f"{scene_dir}/images"
    extrinsics_dir = f"{scene_dir}/extrinsics"
    intrinsics_dir = f"{scene_dir}/intrinsics"
    instances_dir = f"{scene_dir}/instances"
    for p in (images_dir, extrinsics_dir, intrinsics_dir, instances_dir):
        assert os.path.exists(p), f"Missing path: {p}"

    save_dir = f"{scene_dir}/humanpose/temp/Pedes_GTTracks"
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        video_dir = f"{save_dir}/vis"
        os.makedirs(video_dir, exist_ok=True)
        per_human_img_dir = f"{video_dir}/images"
        os.makedirs(per_human_img_dir, exist_ok=True)

    frame_infos = json.load(open(f"{instances_dir}/frame_instances.json"))
    instances_meta = json.load(open(f"{instances_dir}/instances_info.json"))

    collector_all = {}
    for cam_id in camera_list:
        pkl_path = os.path.join(save_dir, f"{cam_id}.pkl")
        if os.path.exists(pkl_path):
            collector_all[cam_id] = json.load(open(pkl_path))
            logger.info(f"Results for camera {cam_id} already exists at {pkl_path}")
            continue

        if verbose:
            per_cam_vis_dir = os.path.join(save_dir, "vis", "images", f"{cam_id}")
            os.makedirs(per_cam_vis_dir, exist_ok=True)

        collector = {}
        frames = []
        for frame_id, frame_ins_list in frame_infos.items():
            frame_id = int(frame_id)

            frame_collector = {
                "gt_bbox": [],
                "extra_data": {"gt_track_id": [], "gt_class": []},
            }

            # Load cam->world for this (frame, cam)
            cam2world = np.loadtxt(
                os.path.join(extrinsics_dir, f"{frame_id:03d}_{cam_id}.txt")
            ).astype(np.float32)
            cam2world = cam2world @ OPENCV2DATASET

            # Load intrinsics
            Ks = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt")).astype(
                np.float32
            )
            fx, fy, cx, cy = Ks[0], Ks[1], Ks[2], Ks[3]
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            # Load image
            ori_image = cv2.imread(os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.jpg"))
            if ori_image is None:
                # Try png fallback
                ori_image = cv2.imread(
                    os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.png")
                )
            if ori_image is None:
                raise FileNotFoundError(
                    f"Missing image for frame={frame_id:03d}, cam={cam_id}"
                )
            image = ori_image.copy()
            H, W = image.shape[:2]

            if len(frame_ins_list) > 0:
                image_plotted = image.copy()
                for instance_id in frame_ins_list:
                    ins = instances_meta[str(instance_id)]
                    if ins["class_name"] not in SMPLNODE_CLASSES:
                        continue

                    ins_anno = ins["frame_annotations"]
                    index = ins_anno["frame_idx"].index(frame_id)
                    obj_to_world = np.array(ins_anno["obj_to_world"][index], dtype=np.float32)
                    l, w, h = ins_anno["box_size"][index]

                    corners = get_corners(l, w, h)  # 3x8
                    corners_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                    world2cam = np.linalg.inv(cam2world)
                    corners_cam = world2cam[:3, :3] @ corners_world + world2cam[:3, 3:4]
                    cam_points, depth = project_camera_points_to_image(corners_cam.T, intrinsic)

                    x_min, y_min = np.min(cam_points, axis=0)
                    x_max, y_max = np.max(cam_points, axis=0)
                    if narrow_width_ratio > 0.0:
                        length = x_max - x_min
                        x_min += length * narrow_width_ratio
                        x_max -= length * narrow_width_ratio

                    original_area = (x_max - x_min) * (y_max - y_min)
                    x_min, x_max = np.clip(x_min, 0, W), np.clip(x_max, 0, W)
                    y_min, y_max = np.clip(y_min, 0, H), np.clip(y_max, 0, H)
                    new_area = (x_max - x_min) * (y_max - y_min)

                    behind = depth.max() < 0
                    too_small = new_area < W * H * (0.03) ** 2
                    too_large = new_area > W * H / 1.1
                    too_far = (
                        np.linalg.norm(obj_to_world[:3, 3] - cam2world[:3, 3]) > 40
                    )
                    clip_large = new_area / original_area < 1 / 3
                    if too_small or too_large or clip_large or behind or too_far:
                        continue

                    gt_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    frame_collector["gt_bbox"].append(gt_box)
                    frame_collector["extra_data"]["gt_track_id"].append(instance_id)
                    frame_collector["extra_data"]["gt_class"].append([0])

                    if verbose:
                        raw_image = cv2.rectangle(
                            ori_image.copy(),
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 255, 0),
                            2,
                        )
                        raw_image_path = os.path.join(
                            per_cam_vis_dir, f"{frame_id}_{instance_id}.jpg"
                        )
                        cv2.imwrite(raw_image_path, raw_image)

                        image_plotted = cv2.rectangle(
                            image,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 255, 0),
                            2,
                        )

                if verbose:
                    frames.append(image_plotted)
            else:
                if verbose:
                    frames.append(ori_image)

            collector[frame_id] = frame_collector

        if verbose and frames:
            height, width = frames[0].shape[:2]
            output_path = os.path.join(save_dir, "vis", f"cam_{cam_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in tqdm(frames, desc=f"Writing video for camera {cam_id}"):
                out.write(frame)
            out.release()

        if save_temp:
            json.dump(collector, open(pkl_path, "w"))

        collector_all[cam_id] = collector

    return collector_all

