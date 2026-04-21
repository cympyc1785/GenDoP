"""Caption all trajectories."""

import os
import glob
import json
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import math
import random 

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed

from helper.files import save_txt, load_json, save_json
from helper.progress import PROGRESS
from processing.captioning import caption_trajectories, get_chunk_descriptions
from processing.segmentation import segment_rigidbody_trajectories


# ------------------------------------------------------------------------------------- #

def load_traj(traj_path):
    traj = []
    with open(traj_path, "r") as f:
        frames = json.load(f)
        for frame in frames:
            R_w2c = torch.tensor(frame['rotation'])
            t_w2c = torch.tensor(frame['position'])
            
            R_c2w = R_w2c.T
            t_c2w = (-R_w2c.T @ t_w2c.unsqueeze(-1)).squeeze(-1)

            c2w = torch.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = t_c2w

            c2w[:3, 1:3] *= -1
            
            # w2c = torch.inverse(c2w)

            traj.append(c2w.tolist())
    
    return torch.tensor(traj)

def tag_segments(traj, scene_dir, config):
    # Segment camera trajectory (no character trajectory)
    cam_segments = segment_rigidbody_trajectories(
        traj,
        cam_static_threshold=config.cam.static_threshold,
        cam_diff_threshold=config.cam.diff_threshold,
        angular_static_threshold=config.cam.angular_static_threshold,
        fps=config.fps,
        min_chunk_size=config.min_chunk_size,
        smoothing_window_size=config.smoothing_window_size,
    )
    
    cam_descriptions = get_chunk_descriptions(cam_segments)
    ref_vid_path = os.path.join(scene_dir, "viz", "vid_49frame.mp4")

    cam_tags = {
        "segments": json.dumps([int(seg) for seg in cam_segments], separators=(',', ':')),
        "description": cam_descriptions
    }
    
    return cam_segments, cam_tags

def caption_tags(cam_segments, traj_name, frame_len, config):
    # Infer the caption with LLM
    model_name="Qwen3-30B-A3B-Instruct-2507"
    start_time = time.time()
    
    caption = caption_trajectories(
        cam_segments=cam_segments,
        context_prompt=config.llm.context,
        instruction_prompt=config.llm.instruction,
        constraint_prompt=config.llm.constraint,
        demonstration_prompt=config.llm.demonstration,
        model_name=model_name,
        shuffle_taxonomy=config["shuffle_taxonomy"],
    )

    result = {}
    inference_type = "camera"
    result[f"prompt_{inference_type}"] = caption
    result[f"metadata_{inference_type}"] = {
        "scene_name": traj_name,
        "total_frames_used": frame_len,
        "model": model_name,
        "analysis_time": datetime.now().isoformat(),
        "inference_time": time.time() - start_time
    }
    
    return result

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

def process_one_scene(task, EXPORT_DIR, config):
    VID_SEG_LEN = 49
    ERR_PATH = os.path.join(EXPORT_DIR, "logs/errors.txt")
    scene_dir, split, scene_name = task
    # data_dir = os.path.join(ROOT_DIR, split)
    # scene_dir = os.path.join(data_dir, scene_name)
    # images_dir = os.path.join(scene_dir, "images")
    images_dir = os.path.join(scene_dir, "rgb")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(scene_dir, "images")
    traj_path = os.path.join(scene_dir, "cameras.json")
    cam_seg_tags_path = os.path.join(EXPORT_DIR, split, scene_name, "tags", "camera_tags_per_seg.json")

    os.makedirs(os.path.dirname(ERR_PATH), exist_ok=True)

    traj_all = load_traj(traj_path)
    

    cam_tags_per_seg = {}
    if os.path.exists(cam_seg_tags_path):
        with open(cam_seg_tags_path, "r", encoding="utf-8") as f:
            cam_tags_per_seg = json.load(f)
    else:
        os.makedirs(os.path.dirname(cam_seg_tags_path), exist_ok=True)

    # Find discontinuous image sequence
    image_filenames = sorted(os.listdir(images_dir))
    if image_filenames[0].startswith("frame"):
        img_num_list = [int(img_filename.split('_')[1].split('.')[0]) for img_filename in image_filenames]
    else:
        img_num_list = [int(img_filename.split('.')[0]) for img_filename in image_filenames]

    _, segments_idx = find_continuous_segments(img_num_list, segment_len=config["segment_length"], use_remaining_frames=False)

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments_idx):
        seg_idx_str = str(seg_idx)
        traj = traj_all[seg_start_idx: seg_end_idx]
        
        try:
            # ===== Tagging =====
            if (not config.overwrite) and (seg_idx_str in cam_tags_per_seg):
                print("Skipping tagging and loading saved tags...")
                data = cam_tags_per_seg[seg_idx_str]
                cam_segments = [int(seg) for seg in data["segments"].strip('[').strip(']').split(',')]
            else:
                cam_segments, cam_tags = tag_segments(traj, scene_dir, config)
                cam_tags_per_seg[seg_idx_str] = cam_tags

                with open(cam_seg_tags_path, "w") as f:
                    json.dump(cam_tags_per_seg, f, indent=4)

            # ===== Caption =====
            if not config["only_tagging"]:
                result = caption_tags(cam_segments, scene_name, VID_SEG_LEN, config)

                if "Error" in result[f"prompt_camera"] or "error" in result[f"prompt_camera"]:
                    with open(ERR_PATH, "a+") as f:
                        f.write(f"{split}/{scene_name}\n")

                save_data = {"frame_idx":[seg_start_idx, seg_end_idx]}
                save_data.update(result)
                
                saved_prompt_path = os.path.join(EXPORT_DIR, split, scene_name, "prompts.json")
                if not os.path.exists(saved_prompt_path):
                    os.makedirs(os.path.dirname(saved_prompt_path), exist_ok=True)
                    with open(saved_prompt_path, "w") as f:
                        json.dump({}, f)
                saved_output = load_json(saved_prompt_path)
                if seg_idx_str in saved_output.keys():
                    saved_output[seg_idx_str].update(save_data)
                else:
                    saved_output[seg_idx_str] = save_data
                save_json(saved_output, saved_prompt_path)
        except Exception as e:
            with open(ERR_PATH, "a+") as f:
                f.write(f"{split}/{scene_name} {seg_idx} ({seg_start_idx},{seg_end_idx}) segment process error {e}\n")


def run_parallel_scenes(worker, tasks, EXPORT_DIR, config, num_workers=1):
    print("Registered Tasks:", len(tasks))

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(worker, t, EXPORT_DIR, config) for t in tasks]

        pbar = tqdm(total=len(futures), desc="Scenes")
        try:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    if fut.done():
                        if fut.exception() is not None:
                            print("FAILED:", fut.exception())
                        pbar.update(1)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            for pid, proc in getattr(ex, "_processes", {}).items():
                try:
                    proc.terminate()
                except Exception:
                    pass
            raise
        finally:
            pbar.close()

def run_all_splits(worker, ROOT_DIR, EXPORT_DIR, splits, config):
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        tasks = []
        for scene_name in sorted(os.listdir(data_dir)):
            scene_dir = os.path.join(data_dir, scene_name)
            tasks.append((scene_dir, split, scene_name))

        run_parallel_scenes(worker, tasks, EXPORT_DIR, config, num_workers=4)

def get_process_time(config, sample_num=500):
    from collections import defaultdict
    def get_tasks(root_dir, splits):
        tasks = []
        for split in splits:
            split_dir = os.path.join(root_dir, split)
            for scene_name in sorted(os.listdir(split_dir)):
                scene_dir = os.path.join(split_dir, scene_name)
                tasks.append((scene_dir, split, scene_name))
        return tasks
    
    def get_seg_stat(tasks):
        total_caption_cnt = 0
        for task in tasks:
            scene_dir, split, scene_name = task
            images_dir = os.path.join(scene_dir, "rgb")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(scene_dir, "images")
            if not os.path.exists(images_dir):
                print("no img", images_dir)
                continue
            caption_path = os.path.join(scene_dir, "prompts.json")
            if not os.path.exists(caption_path):
                print("no cap", images_dir)
                continue
            with open(caption_path, "r") as f:
                data = json.load(f)
            seg_len = len(data.keys())
            total_caption_cnt += seg_len
        return total_caption_cnt

    def get_all_tasks():
        root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
        splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "uvo", "youtube_vis"]
        tasks = get_tasks(root_dir, splits)

        root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
        splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
        tasks += get_tasks(root_dir, splits)

        root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
        splits = [f"{i}K" for i in range(1, 8)]
        tasks += get_tasks(root_dir, splits)

        return tasks

    tasks = get_all_tasks()
    seg_num = get_seg_stat(tasks)
    print("Segment Count:", seg_num)

    tasks = random.sample(tasks, sample_num)

    t = time.time()
    EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/dataset/tmp"
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(process_one_scene, t, EXPORT_DIR, config) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass
    avg_process_time = (time.time() - t)/sample_num

    estimated_process_time = avg_process_time * seg_num
    print(estimated_process_time, "sec")
    print(estimated_process_time/3600, "hours")
    print(estimated_process_time/3600/24, "days")

    exit()

@hydra.main(
    version_base=None,
    config_path="./configs/captioning/",
    config_name="caption_cam+char.yaml",
)
def launch_captioning(config: DictConfig):
    print(config)
    # get_process_time(config)

    # ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DynamicVerse"
    # splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]

    

    # ROOT_DIR = "/data1/cympyc1785/caption/GenDoP/DataDoP"
    # splits = ["ours"]

    # # Error Tasks
    # with open("logs/stat_err.txt", "r") as f:
    #     errors = f.readlines()

    # tasks = []
    # for err in errors:
    #     split, scene_name = err.replace("\n", "").split(" ")[0].split("/")
    #     tasks.append((split, scene_name))
    # run_parallel_scenes(tasks, ROOT_DIR, config, num_workers=4)
    # exit()

    # ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
    # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DL3DV"
    # splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
    # run_all_splits(ROOT_DIR, EXPORT_DIR, splits, config)

    # ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DynamicVerse"
    # splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
    # run_all_splits(ROOT_DIR, EXPORT_DIR, splits, config)

    # ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
    # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DynamicVerse/dynpose-100k"
    # splits = [f"dynpose-{i:04d}" for i in range(0, 50)]
    # run_all_splits(ROOT_DIR, EXPORT_DIR, splits, config)

    # # Add all tasks
    # ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"
    # # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DynamicVerse/dynpose-100k"
    # EXPORT_DIR = "/data1/cympyc1785/caption/GenDoP/OurDataset_tags/DynamicVerse/dynpose-100k"
    # # EXPORT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"
    # # splits = [f"dynpose-{i:04d}" for i in range(67, 73)]
    # # run_all_splits(ROOT_DIR, EXPORT_DIR, splits, config)
    # # run_all_splits(process_one_scene, ROOT_DIR, EXPORT_DIR, splits, config)
    # run_all_splits(process_one_scene_only_tagging, ROOT_DIR, EXPORT_DIR, splits, config)

    # ROOT_DIR = "/data1/cympyc1785/SceneData/tartanair/scenes"
    # EXPORT_DIR = "/data1/cympyc1785/SceneData/tartanair/scenes"
    # splits = sorted(os.listdir(ROOT_DIR))
    # run_all_splits(process_one_scene, ROOT_DIR, EXPORT_DIR, splits, config)

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../WorldTraj/DL3DV")
    EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
    splits = sorted(os.listdir(ROOT_DIR))
    run_all_splits(process_one_scene, ROOT_DIR, EXPORT_DIR, splits, config)

    # tasks = []
    # for err_task in err_tasks:
    #     if len(err_task) < 5:
    #         continue
    #     split, scene_name = err_task.split(" ")[0].split("/")
    #     scene_dir = os.path.join(ROOT_DIR, split, scene_name)
    #     images_dir = os.path.join(scene_dir, "images")
    #     prompts_path = os.path.join(scene_dir, "prompts.json")
    #     camera_path = os.path.join(scene_dir, "cameras.json") # Camera는 연속을 가정

    #     tasks.append((split, scene_name))

    #     # Check minimum validity (cam count == image count)
    #     with open(camera_path, "r") as f:
    #         cameras = json.load(f)
    #     image_filenames = sorted(os.listdir(images_dir))
    #     assert len(image_filenames) == len(cameras)

    #     # # initialize
    #     # with open(prompts_path, "w") as f:
    #     #     json.dump({}, f)
    
    # run_parallel_scenes(tasks, ROOT_DIR, config, num_workers=4)
        

        


if __name__ == "__main__":
    launch_captioning()