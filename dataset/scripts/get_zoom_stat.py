import os
import glob
import json
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import math


import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from helper.files import save_txt, load_json, save_json
from helper.progress import PROGRESS
from processing.captioning import caption_trajectories, get_chunk_descriptions
from processing.segmentation import segment_rigidbody_trajectories

def load_traj_intrinsics(traj_path):
    intrinsics = []
    with open(traj_path, "r") as f:
        frames = json.load(f)
    for frame in frames:
        fx = torch.tensor(frame['fx'])
        intrinsics.append(fx)
    
    return torch.tensor(intrinsics)

def detect_zoom_from_focal(
    focal: torch.Tensor,
    smooth_window: int = 9,
    slope_eps_ratio: float = 0.12,
):
    """
    focal: (T,), (T,2) or (T,*) tensor
        - (T,)  : fx only (or any single focal series)
        - (T,2) : [fx, fy]
        - (T,*) : will use mean over last dim as focal proxy

    smooth_window: odd int, moving-average window
    slope_eps_ratio: "no zoom" threshold as fraction of median focal
                     e.g., 0.01 = 1% of focal per full sequence (trend)
    return:
        label: "zoom in" | "zoom out" | "no zoom"
        info:  dict with slope, slope_norm, noise_sigma (rough)
    """
    assert focal.ndim in (1, 2), "focal must be (T,) or (T,D)"
    x = focal.float()

    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)  # combine fx,fy robustly
    T = x.numel()
    if T < 3:
        return "no zoom", {"slope": 0.0, "slope_norm": 0.0, "noise_sigma": 0.0}

    # --- smoothing (moving average) ---
    w = int(smooth_window)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    w = min(w, T if T % 2 == 1 else T - 1)
    if w >= 3:
        pad = w // 2
        xp = torch.nn.functional.pad(x[None, None, :], (pad, pad), mode="replicate")
        kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / w
        xs = torch.nn.functional.conv1d(xp, kernel).squeeze()
    else:
        xs = x

    # --- linear trend (least squares) ---
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    t_mean = t.mean()
    xs_mean = xs.mean()
    denom = ((t - t_mean) ** 2).sum().clamp_min(1e-12)
    slope = (((t - t_mean) * (xs - xs_mean)).sum() / denom)  # focal units per frame

    # --- normalize slope to be scale-invariant ---
    focal_med = xs.median().clamp_min(1e-12)
    slope_norm = slope * (T - 1) / focal_med  # approx total % change over sequence

    # --- rough noise estimate (MAD of residuals) ---
    resid = xs - (slope * t + (xs_mean - slope * t_mean))
    mad = (resid - resid.median()).abs().median()
    noise_sigma = (1.4826 * mad).item()  # robust sigma-like

    eps = slope_eps_ratio
    if slope_norm.item() > eps:
        label = "zoom in"
    elif slope_norm.item() < -eps:
        label = "zoom out"
    else:
        label = "no zoom"

    info = {
        "slope": float(slope.item()),
        "slope_norm": float(slope_norm.item()),
        "noise_sigma": float(noise_sigma),
        "smooth_window": int(w),
        "threshold_ratio": float(eps),
    }
    return label, info

def find_continuous_segments(nums: list, segment_len=49, use_remaining_frames=False):
    """
    Returns:
        segments_num  : list[(start_num, end_num_exclusive)]
        segments_idx  : list[(start_idx, end_idx_exclusive)]

    Usage:
        data[start_idx:end_idx]
    """
    nums = sorted(nums)

    segments_num = []
    segments_idx = []

    start_num = nums[0]
    start_idx = 0

    prev_num = nums[0]
    length = 1

    for i in range(1, len(nums)):
        num = nums[i]

        if num == prev_num + 1:
            length += 1
        else:
            start_num = num
            start_idx = i
            length = 1

        if length == segment_len:
            segments_num.append((start_num, num + 1))
            segments_idx.append((start_idx, i + 1))

            # reset for non-overlapping segments
            start_num = num + 1
            start_idx = i + 1
            length = 0

        prev_num = num
    
    if use_remaining_frames and length < segment_len and length > 10:
        segments_num.append((start_num, num + 1))
        segments_idx.append((start_idx, i + 1))

    return segments_num, segments_idx

def process_one_scene(task, ROOT_DIR):
    VID_SEG_LEN = 49
    ERR_PATH = "logs/errors.txt"
    split, scene_name = task
    data_dir = os.path.join(ROOT_DIR, split)
    scene_dir = os.path.join(data_dir, scene_name)
    traj_path = os.path.join(scene_dir, "cameras.json")
    images_dir = os.path.join(scene_dir, "rgb")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir):
        with open(ERR_PATH, "a+") as f:
            f.write(f"{split}/{scene_name} no images folder\n")
        return

    intrinsics_all = load_traj_intrinsics(traj_path)

    # segments = math.ceil(len(traj_all) / float(VID_SEG_LEN))
    # Find discontinuous image sequence
    image_filenames = sorted(os.listdir(images_dir))
    if image_filenames[0].startswith("frame"):
        img_num_list = [int(img_filename.split('_')[1].split('.')[0]) for img_filename in image_filenames]
    else:
        img_num_list = [int(img_filename.split('.')[0]) for img_filename in image_filenames]
    _, segments_idx = find_continuous_segments(img_num_list, use_remaining_frames=True)
    

    zoom_in_cnt = 0
    zoom_out_cnt = 0
    no_zoom_cnt = 0

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments_idx):
        seg_idx_str = str(seg_idx)
        intrinsics = intrinsics_all[seg_start_idx: seg_end_idx]

        label, info = detect_zoom_from_focal(intrinsics)

        if label == "zoom in":
            zoom_in_cnt += 1
        elif label == "zoom out":
            zoom_out_cnt += 1
        elif label == "no zoom":
            no_zoom_cnt += 1
        
        if label != "no zoom":
            print(label, task)
    
    return zoom_in_cnt, zoom_out_cnt, no_zoom_cnt
    

# ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
# splits = [f"dynpose-{i:04d}" for i in range(0, 50)]

# ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
# splits = [f"{i}K" for i in range(1, 8)]


ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]

total_zoom_in_cnt = 0
total_zoom_out_cnt = 0
total_no_zoom_cnt = 0

for split in splits:
    print("processing", split)
    data_dir = os.path.join(ROOT_DIR, split)
    for scene_name in tqdm(sorted(os.listdir(data_dir))):
        zoom_in_cnt, zoom_out_cnt, no_zoom_cnt = process_one_scene((split, scene_name), ROOT_DIR)
        total_zoom_in_cnt += zoom_in_cnt
        total_zoom_out_cnt += zoom_out_cnt
        total_no_zoom_cnt += no_zoom_cnt

print("Zoom In", total_zoom_in_cnt)
print("Zoom Out", total_zoom_out_cnt)
print("No Zoom", total_no_zoom_cnt)
    