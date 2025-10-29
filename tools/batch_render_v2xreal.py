#!/usr/bin/env python3

import os
import re
import sys
import glob
import argparse
import subprocess
import logging
from typing import Dict, List

from omegaconf import OmegaConf


log = logging.getLogger("batch_render_v2xreal")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_idx_to_raw(commands_file: str) -> Dict[int, str]:
    """Parse COMMANDS_V2XREAL_CLOSE to map scene_idx -> raw scenario dir."""
    if not os.path.isfile(commands_file):
        raise FileNotFoundError(f"commands_file not found: {commands_file}")
    idx2raw: Dict[int, str] = {}
    with open(commands_file, "r") as f:
        for line in f:
            if ("--split" in line) and ("--scenario" in line) and ("--scene_idx" in line):
                m_split = re.search(r"--split\s+(\S+)", line)
                m_scn = re.search(r"--scenario\s+(\S+)", line)
                m_idx = re.search(r"--scene_idx\s+(\d+)", line)
                if not (m_split and m_scn and m_idx):
                    continue
                split = m_split.group(1)
                scenario = m_scn.group(1)
                idx = int(m_idx.group(1))
                idx2raw[idx] = os.path.join("data", "v2x-real", "raw", split, scenario)
    return idx2raw


def list_runs(runs_root: str, pattern: str) -> List[str]:
    patt = os.path.join(runs_root, pattern)
    runs = [d for d in glob.glob(patt) if os.path.isdir(d)]
    runs.sort()
    return runs


def main():
    ap = argparse.ArgumentParser("Batch render v2x-real runs with overlay")
    ap.add_argument("--runs_root", default="work_dirs/drivestudio", help="Root dir of runs")
    ap.add_argument("--pattern", default="*", help="Glob pattern under runs_root")
    ap.add_argument("--commands_file", default="COMMANDS_V2XREAL_CLOSE", help="File to parse scene_idx->raw mapping")
    ap.add_argument("--render_script", default="tools/render_v2xreal_overlay.py", help="Renderer script path")
    # overlay defaults
    ap.add_argument("--cam_forward_offset", type=float, default=1.0)
    ap.add_argument("--steer_boost", type=float, default=3.0)
    ap.add_argument("--side_cams", type=int, nargs=3, default=[2, 3, 1])
    ap.add_argument("--side_labels", type=str, nargs=3, default=["left", "back", "right"])
    # exec flags
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    idx2raw = parse_idx_to_raw(args.commands_file)
    if not idx2raw:
        log.warning("No scene mapping parsed from commands file; nothing to do.")

    runs = list_runs(args.runs_root, args.pattern)
    if not runs:
        log.warning(f"No runs found under {args.runs_root} with pattern {args.pattern}")
        return

    succ, skip = 0, 0
    for run in runs:
        cfg_path = os.path.join(run, "config.yaml")
        if not os.path.isfile(cfg_path):
            log.debug(f"skip (no config): {run}")
            continue
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception as e:
            log.warning(f"skip (bad config): {run}: {e}")
            continue
        data = cfg.get("data", {})
        dataset = str(data.get("dataset", "")).lower()
        if dataset != "v2xreal":
            log.debug(f"skip (dataset={dataset}): {run}")
            continue
        scene_idx = data.get("scene_idx", None)
        if scene_idx is None:
            log.warning(f"skip (no scene_idx): {run}")
            continue
        raw_dir = idx2raw.get(int(scene_idx))
        if not raw_dir:
            log.warning(f"skip (no raw mapping for scene_idx={scene_idx}): {run}")
            continue
        cmd = [
            sys.executable, args.render_script,
            "--run_dir", run,
            "--raw_scenario_dir", raw_dir,
            "--cam_forward_offset", str(args.cam_forward_offset),
            "--steer_boost", str(args.steer_boost),
            "--side_cams", str(args.side_cams[0]), str(args.side_cams[1]), str(args.side_cams[2]),
            "--side_labels", args.side_labels[0], args.side_labels[1], args.side_labels[2],
        ]
        print("==>", " ".join(cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
            succ += 1
        except subprocess.CalledProcessError as e:
            log.error(f"render failed: {run}: {e}")
            skip += 1
    log.info(f"Done. success={succ}, failed={skip}, total={succ+skip}")


if __name__ == "__main__":
    main()

