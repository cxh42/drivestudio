#!/usr/bin/env python3
"""
Create a subsampled copy of a processed scene directory (e.g., V2X-Real/NuScenes-style),
keeping every N-th frame with an optional offset. This is useful to build a 5Hz
dataset from a 10Hz scene while keeping file naming and indices contiguous.

It copies:
  - images/{frame}_{cam}.(jpg|png)
  - extrinsics/{frame}_{cam}.txt
  - lidar/{frame}.bin (if exists)
  - lidar_pose/{frame}.txt (if exists)
  - intrinsics/*.txt (as-is)
  - instances/{instances_info.json, frame_instances.json} with frame indices remapped
  - optional masks if present: dynamic_masks/*, sky_masks/* (best-effort)

Usage:
  python datasets/tools/subsample_processed_scene.py \
      --source data/v2x-real/processed_multicam/002 \
      --target data/v2x-real/processed_multicam_5Hz/002 \
      --factor 2 --offset 0
"""

from __future__ import annotations

import os
import re
import json
import shutil
import argparse
from typing import List, Dict, Tuple


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_frames(images_dir: str) -> List[int]:
    frames = set()
    for name in os.listdir(images_dir):
        m = re.match(r"^(\d{3})_\d+\.(?:jpg|jpeg|png|JPG|JPEG|PNG)$", name)
        if m:
            frames.add(int(m.group(1)))
    return sorted(frames)


def list_cams(intrinsics_dir: str) -> List[int]:
    cams = []
    for name in os.listdir(intrinsics_dir):
        if name.endswith(".txt") and name[:-4].isdigit():
            cams.append(int(name[:-4]))
    return sorted(cams)


def copy_intrinsics(src: str, dst: str) -> None:
    sdir = os.path.join(src, "intrinsics")
    ddir = os.path.join(dst, "intrinsics")
    if os.path.isdir(sdir):
        ensure_dir(ddir)
        for f in os.listdir(sdir):
            shutil.copy2(os.path.join(sdir, f), os.path.join(ddir, f))


def copy_masks(src: str, dst: str, kept_old_to_new: Dict[int, int]) -> None:
    # dynamic masks: support both dynamic_masks and fine_dynamic_masks
    for dyn_root in ["fine_dynamic_masks", "dynamic_masks"]:
        dyn_dir = os.path.join(src, dyn_root)
        if not os.path.isdir(dyn_dir):
            continue
        for sub in ["all", "human", "vehicle"]:
            sdir = os.path.join(dyn_dir, sub)
            if not os.path.isdir(sdir):
                continue
            ddir = os.path.join(dst, dyn_root, sub)
            ensure_dir(ddir)
            for name in os.listdir(sdir):
                m = re.match(r"^(\d{3})_(\d+)\.(?:png|PNG)$", name)
                if not m:
                    continue
                old_idx = int(m.group(1))
                if old_idx not in kept_old_to_new:
                    continue
                new_idx = kept_old_to_new[old_idx]
                new_name = f"{new_idx:03d}_{m.group(2)}.png"
                shutil.copy2(os.path.join(sdir, name), os.path.join(ddir, new_name))

    # sky masks: sky_masks/{frame}_{cam}.png
    sky_dir = os.path.join(src, "sky_masks")
    if os.path.isdir(sky_dir):
        ddir = os.path.join(dst, "sky_masks")
        ensure_dir(ddir)
        for name in os.listdir(sky_dir):
            m = re.match(r"^(\d{3})_(\d+)\.(?:png|PNG)$", name)
            if not m:
                continue
            old_idx = int(m.group(1))
            if old_idx not in kept_old_to_new:
                continue
            new_idx = kept_old_to_new[old_idx]
            new_name = f"{new_idx:03d}_{m.group(2)}.png"
            shutil.copy2(os.path.join(sky_dir, name), os.path.join(ddir, new_name))


def copy_images_and_extrinsics(src: str, dst: str, cams: List[int], kept_old_to_new: Dict[int, int]) -> None:
    # images
    sdir = os.path.join(src, "images")
    ddir = os.path.join(dst, "images")
    ensure_dir(ddir)
    for old_idx, new_idx in kept_old_to_new.items():
        for cam in cams:
            for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
                src_img = os.path.join(sdir, f"{old_idx:03d}_{cam}.{ext}")
                if os.path.exists(src_img):
                    dst_img = os.path.join(ddir, f"{new_idx:03d}_{cam}.{ext.lower()}")
                    shutil.copy2(src_img, dst_img)
                    break
    # extrinsics
    sdir = os.path.join(src, "extrinsics")
    ddir = os.path.join(dst, "extrinsics")
    ensure_dir(ddir)
    for old_idx, new_idx in kept_old_to_new.items():
        for cam in cams:
            src_ex = os.path.join(sdir, f"{old_idx:03d}_{cam}.txt")
            if os.path.exists(src_ex):
                dst_ex = os.path.join(ddir, f"{new_idx:03d}_{cam}.txt")
                shutil.copy2(src_ex, dst_ex)


def copy_lidar(src: str, dst: str, kept_old_to_new: Dict[int, int]) -> None:
    # lidar
    for sub in ("lidar", "lidar_pose"):
        sdir = os.path.join(src, sub)
        if not os.path.isdir(sdir):
            continue
        ddir = os.path.join(dst, sub)
        ensure_dir(ddir)
        for old_idx, new_idx in kept_old_to_new.items():
            if sub == "lidar":
                src_f = os.path.join(sdir, f"{old_idx:03d}.bin")
                dst_f = os.path.join(ddir, f"{new_idx:03d}.bin")
            else:
                src_f = os.path.join(sdir, f"{old_idx:03d}.txt")
                dst_f = os.path.join(ddir, f"{new_idx:03d}.txt")
            if os.path.exists(src_f):
                shutil.copy2(src_f, dst_f)


def remap_instances(src: str, dst: str, kept_old_to_new: Dict[int, int]) -> None:
    idir = os.path.join(src, "instances")
    if not os.path.isdir(idir):
        return
    odir = os.path.join(dst, "instances")
    ensure_dir(odir)
    inst_info_p = os.path.join(idir, "instances_info.json")
    frame_inst_p = os.path.join(idir, "frame_instances.json")
    if not (os.path.exists(inst_info_p) and os.path.exists(frame_inst_p)):
        return

    instances_info = json.load(open(inst_info_p, "r"))
    frame_instances = json.load(open(frame_inst_p, "r"))

    # Remap frame_instances
    new_frame_instances: Dict[str, List[int]] = {}
    for old_idx, new_idx in kept_old_to_new.items():
        key = str(old_idx)
        if key in frame_instances:
            new_frame_instances[str(new_idx)] = frame_instances[key]

    # Remap per-instance annotations
    new_instances_info = {}
    for k, meta in instances_info.items():
        frames = meta["frame_annotations"]["frame_idx"]
        T_list = meta["frame_annotations"]["obj_to_world"]
        size_list = meta["frame_annotations"]["box_size"]
        sel_frames: List[int] = []
        sel_T: List = []
        sel_size: List = []
        for idx, f in enumerate(frames):
            if f in kept_old_to_new:
                sel_frames.append(kept_old_to_new[f])
                sel_T.append(T_list[idx])
                sel_size.append(size_list[idx])
        new_meta = dict(meta)
        new_meta["frame_annotations"] = {
            "frame_idx": sel_frames,
            "obj_to_world": sel_T,
            "box_size": sel_size,
        }
        new_instances_info[k] = new_meta

    json.dump(new_instances_info, open(os.path.join(odir, "instances_info.json"), "w"))
    json.dump(new_frame_instances, open(os.path.join(odir, "frame_instances.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description="Subsample a processed multi-cam scene by factor with offset")
    ap.add_argument("--source", required=True, help="Source scene dir (e.g., .../processed_multicam/002)")
    ap.add_argument("--target", required=True, help="Target scene dir (e.g., .../processed_multicam_5Hz/002)")
    ap.add_argument("--factor", type=int, default=2, help="Keep every N-th frame")
    ap.add_argument("--offset", type=int, default=0, help="Start offset in [0, N-1]")
    args = ap.parse_args()

    src = args.source
    dst = args.target
    assert os.path.isdir(src), f"Source not found: {src}"
    ensure_dir(dst)

    images_dir = os.path.join(src, "images")
    intr_dir = os.path.join(src, "intrinsics")
    assert os.path.isdir(images_dir) and os.path.isdir(intr_dir), "Missing images/intrinsics in source scene"

    cams = list_cams(intr_dir)
    all_frames = list_frames(images_dir)
    kept = [f for f in all_frames if (f - all_frames[0]) % args.factor == args.offset]
    if not kept:
        raise RuntimeError("No frames selected; check factor/offset.")

    kept_old_to_new: Dict[int, int] = {old: i for i, old in enumerate(kept)}

    copy_intrinsics(src, dst)
    copy_images_and_extrinsics(src, dst, cams, kept_old_to_new)
    copy_lidar(src, dst, kept_old_to_new)
    copy_masks(src, dst, kept_old_to_new)
    remap_instances(src, dst, kept_old_to_new)

    print(f"Done. Subsampled {len(kept)} / {len(all_frames)} frames -> {dst}")


if __name__ == "__main__":
    main()
