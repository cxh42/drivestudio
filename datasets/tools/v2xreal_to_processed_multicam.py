import os
import glob
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict:
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def x_to_world(pose6: List[float]) -> np.ndarray:
    x, y, z, roll, yaw, pitch = pose6[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    T[0, 0] = c_p * c_y
    T[0, 1] = c_y * s_p * s_r - s_y * c_r
    T[0, 2] = -c_y * s_p * c_r - s_y * s_r
    T[1, 0] = s_y * c_p
    T[1, 1] = s_y * s_p * s_r + c_y * c_r
    T[1, 2] = -s_y * s_p * c_r + c_y * s_r
    T[2, 0] = s_p
    T[2, 1] = -c_p * s_r
    T[2, 2] = c_p * c_r
    return T


import re

CAM_KEYS_CANDIDATES = [
    ["cam1_left", "cam2_left", "cam3_left", "cam4_left"],
    ["cam1", "cam2", "cam3", "cam4"],
]


def extract_cam_keys(meta: Dict) -> List[str]:
    """Return available camera keys in order (cam1..cam4 or cam*_left variants).
    Fallback: regex match keys like 'cam<idx>' optionally with suffixes.
    """
    # Try known candidates preserving order
    for cand in CAM_KEYS_CANDIDATES:
        if all(k in meta for k in cand):
            return cand
    # Collect any keys matching cam[1-9]
    found: List[Tuple[int, str]] = []
    for k in meta.keys():
        m = re.match(r"^cam(\d+)(?:_.*)?$", k)
        if m and isinstance(meta[k], dict) and 'intrinsic' in meta[k] and 'extrinsic' in meta[k]:
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            found.append((idx, k))
    found.sort(key=lambda x: x[0])
    return [k for _, k in found]


def cam_key_to_num(cam_key: str) -> int:
    m = re.match(r"^cam(\d+)(?:_.*)?$", cam_key)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return -1


def list_frame_ids(agent_dir: str) -> List[str]:
    ylist = sorted(glob.glob(os.path.join(agent_dir, '*.yaml')))
    return [os.path.splitext(os.path.basename(p))[0] for p in ylist]


def discover_image(agent_dir: str, frame_id: str, cam_num: int) -> str:
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        p = os.path.join(agent_dir, f"{frame_id}_cam{cam_num}.{ext}")
        if os.path.exists(p):
            return p
    # fallback: any image with this prefix
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        pats = glob.glob(os.path.join(agent_dir, f"{frame_id}_*.{ext}"))
        if pats:
            return sorted(pats)[0]
    return ""


VEHICLE_CLS = {"car", "van", "bus", "truck", "trailer", "tram", "suv", "mpv"}
HUMAN_CLS = {"pedestrian", "person", "cyclist", "rider"}


def map_class(obj_type: str) -> str:
    t = obj_type.lower().strip()
    if t in VEHICLE_CLS:
        return 'vehicle.car'
    if t in HUMAN_CLS:
        return 'human.pedestrian.adult'
    return 'vehicle.car'


def main():
    ap = argparse.ArgumentParser(description='Convert a V2X-Real scenario to a single multi-camera processed scene (vehicles + infrastructures)')
    ap.add_argument('--source_root', default='data/v2x-real/raw')
    ap.add_argument('--split', default='validate', choices=['train', 'validate', 'test'])
    ap.add_argument('--scenario', required=True, help='Scenario folder name under split')
    ap.add_argument('--vehicle_id', default=None, help='[Deprecated] Single vehicle agent id (e.g., 1); use --vehicle_ids instead')
    ap.add_argument('--vehicle_ids', default=None, nargs='+', help='One or more vehicle agent folder names, e.g., 1 2')
    ap.add_argument('--infra_id', default=None, help='[Deprecated] Single infrastructure agent id (e.g., -1); use --infra_ids instead')
    ap.add_argument('--infra_ids', default=None, nargs='+', help='One or more infrastructure agent folder names, e.g., -1 -2')
    ap.add_argument('--target_root', default='data/v2x-real/processed_multicam')
    ap.add_argument('--scene_idx', type=int, default=0)
    ap.add_argument('--instances_source', default='vehicle', choices=['vehicle', 'infra', 'union'])
    ap.add_argument('--skip_images', action='store_true')
    ap.add_argument('--expected_cams', type=int, default=None, help='Sanity check: require total camera count == expected_cams')
    # LiDAR merge options
    ap.add_argument('--merge_vehicle_lidar', action='store_true',
                    help='Merge additional vehicle LiDARs into the main LiDAR (veh0) to improve coverage')
    ap.add_argument('--merge_lidar_max_points', type=int, default=200000,
                    help='Optional cap on merged LiDAR points per frame in main LiDAR local frame')
    args = ap.parse_args()

    scn_dir = os.path.join(args.source_root, args.split, args.scenario)
    vehicle_ids: List[str]
    if args.vehicle_ids is not None:
        vehicle_ids = args.vehicle_ids
    elif args.vehicle_id is not None:
        vehicle_ids = [args.vehicle_id]
    else:
        vehicle_ids = []
    assert len(vehicle_ids) > 0, 'At least one vehicle id is required (use --vehicle_ids).'
    vehicle_dirs: List[Tuple[str, str]] = []
    for vid in vehicle_ids:
        d = os.path.join(scn_dir, vid)
        if not os.path.isdir(d):
            raise RuntimeError(f'Missing vehicle agent dir: {d}')
        vehicle_dirs.append((vid, d))

    infra_ids: List[str]
    if args.infra_ids is not None:
        infra_ids = args.infra_ids
    elif args.infra_id is not None:
        infra_ids = [args.infra_id]
    else:
        infra_ids = []
    infra_dirs = []
    for iid in infra_ids:
        d = os.path.join(scn_dir, iid)
        if not os.path.isdir(d):
            print(f'[v2xreal_to_nuscenes_8cams] Warning: infra agent dir missing: {d}, skip.')
            continue
        infra_dirs.append((iid, d))

    out_scene = os.path.join(args.target_root, f"{args.scene_idx:03d}")
    ensure_dir(os.path.join(out_scene, 'images'))
    ensure_dir(os.path.join(out_scene, 'intrinsics'))
    ensure_dir(os.path.join(out_scene, 'extrinsics'))
    ensure_dir(os.path.join(out_scene, 'lidar'))
    ensure_dir(os.path.join(out_scene, 'lidar_pose'))

    # Find common frames (by filename stem)
    # Common frames across all vehicles and infra agents
    common = None
    for _, d in vehicle_dirs:
        frames = set(list_frame_ids(d))
        common = frames if common is None else (common & frames)
    for _, d in infra_dirs:
        common &= set(list_frame_ids(d))
    common_frames = sorted(list(common))
    if len(common_frames) == 0:
        raise RuntimeError('No common frame ids among all selected vehicles and infrastructures.')

    # Determine camera keys from first vehicle YAML
    # For each vehicle agent, extract its cam keys
    vehicle_cam_keys: List[Tuple[str, List[str]]] = []
    for vid, d in vehicle_dirs:
        first_yaml = load_yaml(os.path.join(d, f"{common_frames[0]}.yaml"))
        keys = extract_cam_keys(first_yaml)
        if not keys:
            raise RuntimeError(f'Vehicle {vid} YAML missing camera keys (cam1..cam4).')
        vehicle_cam_keys.append((vid, keys))
    # For each infra agent, extract its cam keys (may be 0, 2 or 4)
    infra_cam_keys: List[Tuple[str, List[str]]] = []
    for iid, d in infra_dirs:
        jy = load_yaml(os.path.join(d, f"{common_frames[0]}.yaml"))
        keys = extract_cam_keys(jy)
        if not keys:
            print(f'[v2xreal_to_nuscenes_8cams] Warning: infra {iid} has no camera keys; skipping cameras.')
        infra_cam_keys.append((iid, keys))

    # Write intrinsics: vehicles first, then infra; number is adaptive
    cam_offset = 0
    total_cams = 0
    for (vid, d), (_, keys) in zip(vehicle_dirs, vehicle_cam_keys):
        first_yaml = load_yaml(os.path.join(d, f"{common_frames[0]}.yaml"))
        for off, cam_key in enumerate(keys):
            cam_idx = cam_offset + off
            intr = np.array(first_yaml[cam_key]['intrinsic'], dtype=np.float64).reshape(3, 3)
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            Ks = np.array([fx, fy, cx, cy, 0., 0., 0., 0., 0.], dtype=np.float64)
            np.savetxt(os.path.join(out_scene, 'intrinsics', f'{cam_idx}.txt'), Ks)
        cam_offset += len(keys)
        total_cams += len(keys)
    for (iid, d), (_, keys) in zip(infra_dirs, infra_cam_keys):
        if not keys:
            continue
        j0 = load_yaml(os.path.join(d, f"{common_frames[0]}.yaml"))
        for off, cam_key in enumerate(keys):
            cam_idx = cam_offset + off
            intr = np.array(j0[cam_key]['intrinsic'], dtype=np.float64).reshape(3, 3)
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            Ks = np.array([fx, fy, cx, cy, 0., 0., 0., 0., 0.], dtype=np.float64)
            np.savetxt(os.path.join(out_scene, 'intrinsics', f'{cam_idx}.txt'), Ks)
        cam_offset += len(keys)
        total_cams += len(keys)

    if args.expected_cams is not None and total_cams != args.expected_cams:
        raise RuntimeError(f"Camera count mismatch: got {total_cams}, expected {args.expected_cams}.")

    # Iterate frames in chronological order
    for out_idx, frame_id in enumerate(common_frames):
        veh_yamls = [(vid, load_yaml(os.path.join(d, f"{frame_id}.yaml"))) for vid, d in vehicle_dirs]
        infra_yamls = [(iid, load_yaml(os.path.join(d, f"{frame_id}.yaml"))) for iid, d in infra_dirs]

        # LiDAR from the first vehicle (single LiDAR source for DriveStudio)
        # Optionally merge other vehicle LiDARs into the main LiDAR local frame.
        vmeta0 = veh_yamls[0][1]
        T_l2w_v0 = x_to_world(vmeta0.get('lidar_pose', vmeta0.get('true_ego_pose')))
        np.savetxt(os.path.join(out_scene, 'lidar_pose', f'{out_idx:03d}.txt'), T_l2w_v0)
        # Load main LiDAR points in veh0 local frame
        main_bin = os.path.join(vehicle_dirs[0][1], f'{frame_id}.bin')
        pts_main = None
        if os.path.exists(main_bin):
            try:
                pts_main = np.fromfile(main_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
            except Exception:
                pts_main = None
        merged_pts = []
        if pts_main is not None:
            merged_pts.append(pts_main)
        # Merge additional vehicle LiDARs
        if args.merge_vehicle_lidar and len(vehicle_dirs) > 1:
            for (vid, d) in vehicle_dirs[1:]:
                # Load veh LiDAR and pose for this frame
                ypath = os.path.join(d, f'{frame_id}.yaml')
                bpath = os.path.join(d, f'{frame_id}.bin')
                if not (os.path.exists(ypath) and os.path.exists(bpath)):
                    continue
                try:
                    y = yaml.safe_load(open(ypath, 'r'))
                    T_l2w = x_to_world(y.get('lidar_pose', y.get('true_ego_pose')))
                    pts = np.fromfile(bpath, dtype=np.float32).reshape(-1, 4)[:, :3]
                    # veh local -> world
                    pts_w = (pts @ T_l2w[:3, :3].T) + T_l2w[:3, 3]
                    # world -> veh0 local (homogeneous)
                    W2L0 = np.linalg.inv(T_l2w_v0)
                    pts_l0 = (np.hstack([pts_w, np.ones((len(pts_w), 1))]) @ W2L0[:3, :].T)
                    merged_pts.append(pts_l0)
                except Exception:
                    continue
        # Concatenate and optionally cap, then write to processed lidar/{frame}.bin
        if merged_pts:
            all_pts = np.vstack(merged_pts).astype(np.float32)
            if args.merge_lidar_max_points and args.merge_lidar_max_points > 0 and len(all_pts) > args.merge_lidar_max_points:
                sel = np.random.choice(len(all_pts), args.merge_lidar_max_points, replace=False)
                all_pts = all_pts[sel]
            out = np.zeros((len(all_pts), 4), dtype=np.float32)
            out[:, :3] = all_pts
            out_path = os.path.join(out_scene, 'lidar', f'{out_idx:03d}.bin')
            out.tofile(out_path)
        else:
            # fallback: copy main if available, else create empty
            out_path = os.path.join(out_scene, 'lidar', f'{out_idx:03d}.bin')
            if os.path.exists(main_bin):
                with open(main_bin, 'rb') as fsrc, open(out_path, 'wb') as fdst:
                    fdst.write(fsrc.read())
            else:
                open(out_path, 'ab').close()
        # Vehicle cameras: for each vehicle agent, append its cams
        cam_base = 0
        for (vid, d), (_, keys) in zip(vehicle_dirs, vehicle_cam_keys):
            vmeta = dict([p for p in veh_yamls if p[0] == vid][0][1])
            T_l2w_v = x_to_world(vmeta.get('lidar_pose', vmeta.get('true_ego_pose')))
            for local_cam, cam_key in enumerate(keys):
                cam_idx = cam_base + local_cam
                T_c2l = np.array(vmeta[cam_key]['extrinsic'], dtype=np.float64).reshape(4, 4)
                T_c2w = T_l2w_v @ T_c2l
                np.savetxt(os.path.join(out_scene, 'extrinsics', f'{out_idx:03d}_{cam_idx}.txt'), T_c2w)
                if not args.skip_images:
                    cam_num = cam_key_to_num(cam_key)
                    src = discover_image(d, frame_id, cam_num if cam_num > 0 else (local_cam + 1))
                    if src:
                        dst = os.path.join(out_scene, 'images', f'{out_idx:03d}_{cam_idx}.jpg')
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
            cam_base += len(keys)

        # Infra cameras after vehicle cams: for each infra agent
        for (iid, d), (_, keys) in zip(infra_dirs, infra_cam_keys):
            if not keys:
                continue
            imeta = dict([pair for pair in infra_yamls if pair[0] == iid][0][1])
            T_l2w_i = x_to_world(imeta.get('lidar_pose', imeta.get('true_ego_pose')))
            for local_cam, cam_key in enumerate(keys):
                cam_idx = cam_base + local_cam
                T_c2l = np.array(imeta[cam_key]['extrinsic'], dtype=np.float64).reshape(4, 4)
                T_c2w = T_l2w_i @ T_c2l
                np.savetxt(os.path.join(out_scene, 'extrinsics', f'{out_idx:03d}_{cam_idx}.txt'), T_c2w)
                if not args.skip_images:
                    cam_num = cam_key_to_num(cam_key)
                    src = discover_image(d, frame_id, cam_num if cam_num > 0 else (local_cam + 1))
                    if src:
                        dst = os.path.join(out_scene, 'images', f'{out_idx:03d}_{cam_idx}.jpg')
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
            cam_base += len(keys)

    # Instances (optional union)
    if args.instances_source in ('vehicle', 'infra', 'union'):
        # Build per-frame object dicts
        def parse_frame_objs(yaml_meta: Dict) -> Dict[int, Dict]:
            objs = {}
            vehs = yaml_meta.get('vehicles', {}) or {}
            for k, v in vehs.items():
                try:
                    oid = int(k)
                except Exception:
                    continue
                loc = v.get('location', [0.0, 0.0, 0.0])
                center = np.array([float(loc[0]), float(loc[1]), float(loc[2])], dtype=np.float64)
                ang = v.get('angle', [0.0, 0.0, 0.0])
                yaw = float(ang[1])
                ext = v.get('extent', [0.0, 0.0, 0.0])
                l, w, h = 2.0 * float(ext[0]), 2.0 * float(ext[1]), 2.0 * float(ext[2])
                cls = v.get('obj_type', 'car')
                T = np.eye(4, dtype=np.float64)
                c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                T[:3, 3] = center
                objs[oid] = {'class_name': map_class(cls), 'T_o2w': T.tolist(), 'size_lwh': [l, w, h]}
            return objs

        # Aggregate across frames
        tracks: Dict[int, Dict] = {}
        for out_idx, frame_id in enumerate(common_frames):
            frame_objs = {}
            if args.instances_source in ('vehicle', 'union'):
                for vid, d in vehicle_dirs:
                    vmeta = load_yaml(os.path.join(d, f"{frame_id}.yaml"))
                    for oid, ob in parse_frame_objs(vmeta).items():
                        if oid not in frame_objs:
                            frame_objs[oid] = ob
            if args.instances_source in ('infra', 'union'):
                for iid, d in infra_dirs:
                    imeta = load_yaml(os.path.join(d, f"{frame_id}.yaml"))
                    for oid, ob in parse_frame_objs(imeta).items():
                        if oid not in frame_objs:
                            frame_objs[oid] = ob
            for oid, ob in frame_objs.items():
                tr = tracks.get(oid)
                if tr is None:
                    tracks[oid] = {
                        'id': oid,
                        'class_name': ob['class_name'],
                        'frames': [out_idx],
                        'T_o2w': [ob['T_o2w']],
                        'size_lwh': [ob['size_lwh']],
                    }
                else:
                    tr['frames'].append(out_idx)
                    tr['T_o2w'].append(ob['T_o2w'])
                    tr['size_lwh'].append(ob['size_lwh'])

        # Remap to compact ids 0..N-1
        id_list = sorted(tracks.keys())
        remap = {oid: i for i, oid in enumerate(id_list)}
        instances_info = {}
        frame_instances = {str(i): [] for i in range(len(common_frames))}
        for oid, tr in tracks.items():
            k = remap[oid]
            instances_info[str(k)] = {
                'id': k,
                'class_name': tr['class_name'],
                'frame_annotations': {
                    'frame_idx': tr['frames'],
                    'obj_to_world': tr['T_o2w'],
                    'box_size': tr['size_lwh'],
                }
            }
            for fi in tr['frames']:
                frame_instances[str(fi)].append(k)

        inst_dir = os.path.join(out_scene, 'instances')
        ensure_dir(inst_dir)
        with open(os.path.join(inst_dir, 'instances_info.json'), 'w') as f:
            json.dump(instances_info, f, indent=2)
        with open(os.path.join(inst_dir, 'frame_instances.json'), 'w') as f:
            json.dump(frame_instances, f, indent=2)

    # Final sanity checks: if not skipping images, ensure all images exist
    if not args.skip_images:
        missing = []
        for fi, frame_id in enumerate(common_frames):
            for cam_idx in range(total_cams):
                img = os.path.join(out_scene, 'images', f'{fi:03d}_{cam_idx}.jpg')
                if not os.path.exists(img):
                    missing.append(img)
        if missing:
            raise RuntimeError(f'Missing images ({len(missing)} files). Example: {missing[:5]}')

    print(f'[v2xreal_to_processed_multicam] Done: {out_scene}\n  frames={len(common_frames)}, cams={total_cams}\n  intrinsics={len(os.listdir(os.path.join(out_scene, "intrinsics")))} files\n  images_dir={os.path.join(out_scene, "images")}')


if __name__ == '__main__':
    main()
