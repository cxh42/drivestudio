import argparse
import os
import time
import logging
from omegaconf import OmegaConf

import torch

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser("Open trained model in browser viewer (viser)")
    ap.add_argument("--resume_from", required=True, type=str, help="Path to checkpoint to load (e.g., work_dirs/.../checkpoint_final.pth)")
    ap.add_argument("--viewer_port", type=int, default=8080, help="Viewer port (default: 8080)")
    # Optional overrides to config.yaml, same as eval/train
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides, e.g. data.scene_idx=0")
    args = ap.parse_args()

    ckpt_path = args.resume_from
    log_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(log_dir, "config.yaml")
    assert os.path.exists(cfg_path), f"config.yaml not found in {log_dir}"

    cfg = OmegaConf.load(cfg_path)
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset (for layout, aabb, etc.)
    dataset = DrivingDataset(data_cfg=cfg.data)

    # Build trainer and load checkpoint
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
    logger.info(f"Loaded checkpoint {ckpt_path} at step {trainer.step}")

    # Init viewer
    trainer.init_viewer(port=args.viewer_port)
    url = f"http://localhost:{args.viewer_port}"
    print(f"Viewer running at {url} (Ctrl+C to exit)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Viewer closed.")


if __name__ == "__main__":
    main()

