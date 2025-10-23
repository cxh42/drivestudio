# V2X‑Real → DriveStudio Conversion

This guide summarizes how to convert a V2X‑Real scenario (OPV2V‑like layout) into the DriveStudio processed format so it can be loaded by the NuScenes sourceloader.

Important: the public V2X‑Real you placed under `data/v2x-real/raw` currently contains LiDAR (`.bin`) and YAML metadata, but no camera images. DriveStudio training requires images. The converter below supports exporting LiDAR + calibrations; if you later provide images, it will also copy them.

## Expected Layouts

Raw (example):
- `data/v2x-real/raw/<split>/<scenario>/<ego_id>/` with `*.yaml` and `*.bin`
- YAML keys include `cam1_left`..`cam4_left` with 3×3 intrinsics and 4×4 camera→LiDAR extrinsics, plus `lidar_pose`/`true_ego_pose` (x,y,z,roll,yaw,pitch in deg)

Processed (DriveStudio):
- `<target_root>/<scene_idx>/`
  - `images/{frame}_{cam}.jpg` (if available)
  - `intrinsics/{cam}.txt` (fx,fy,cx,cy,k1..k3=0)
  - `extrinsics/{frame}_{cam}.txt` (camera→world 4×4)
  - `lidar/{frame}.bin` (x,y,z,i float32)
  - `lidar_pose/{frame}.txt` (lidar→world 4×4)
  - `instances/{instances_info.json, frame_instances.json}` (optional; see below)

## Convert One Scenario (Multi‑Agent → One Scene)

一键将多车辆+多设施转为单个可训练场景（相机数自适应，常见 12 路：两车×4 + 两杆×2）：

```
python datasets/tools/v2xreal_to_processed_multicam.py \
  --source_root data/v2x-real/raw \
  --split validate \
  --scenario 2023-03-17-16-03-02_11_1 \
  --vehicle_ids 1 2 \
  --infra_ids -1 -2 \
  --target_root data/v2x-real/processed_multicam \
  --scene_idx 0 \
  --instances_source union \
  --expected_cams 12
```

脚本会自动求公共帧，对所有可用相机导出 `images/`、`intrinsics/`、`extrinsics/`，并用第一辆车的 LiDAR 作为场景 LiDAR 源。

Notes:
- Instances use the per‑frame `vehicles` IDs as stable tracks, and map coarse classes to NuScenes‑style labels (`vehicle.car`, `human.pedestrian.adult`). Box sizes are doubled from `extent` (half sizes) to length‑width‑height.
- If you later add images, re‑run step (1) without `--skip_images` to populate `images/`.

## Sample Config

Use `configs/datasets/v2xreal/12cams.yaml` (12 cameras). Adjust `data_root`, `scene_idx`, and camera list according to your output.

```
python tools/train.py \
  --config_file configs/omnire.yaml \
  dataset=v2xreal/12cams \
  data.data_root=data/v2x-real/processed_multicam \
  data.scene_idx=0
```

Caveats:
- Without images, DriveStudio’s photometric training cannot run. You can still verify loading of LiDAR and instances via the dataset builder.
- The camera names are `cam1_left..cam4_left` with assumed original size (1080, 1920); intrinsics scaling uses this size.

## Reference
- DAIR‑V2X conversion in `RE0/drivestudio` reuses the same NuScenes sourceloader pattern and informed this implementation.
