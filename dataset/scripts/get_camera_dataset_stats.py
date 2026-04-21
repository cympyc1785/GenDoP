"""Caption all trajectories."""

import os
import json
import shutil

from datetime import datetime

from processing.segmentation import segment_rigidbody_trajectories, find_consecutive_chunks, ANG_INDEX_TO_PATTERN, CAM_INDEX_TO_PATTERN

from collections import defaultdict

from tqdm import tqdm

from functools import reduce

# ------------------------------------------------------------------------------------- #
SEG_LEN = 49

def log_err(err_msg):
    with open("logs/stat_err.txt", "a+") as f:
        f.write(err_msg + "\n")

def get_dataset_stat(root_dir, splits, mode="tag"):
    stat_all_data = {"cam":defaultdict(int),
                 "ang":defaultdict(int)}
    caption_stat_data = defaultdict(int)
    binary_stat_data = {"static":0, "dynamic":0}
    split_stat_data = {}
    combined_stat_all_data = defaultdict(int)

    for split in splits:
        if "dynpose-100k" in split:
            continue
        split_stat_data[split] = {"cam":defaultdict(int),
                            "ang":defaultdict(int)}
        print("split", split)
        data_dir = f"{root_dir}/{split}"
        if not os.path.exists(data_dir):
            continue
        for scene_name in sorted(os.listdir(data_dir)):
            if scene_name == "backup":
                continue
            src_dir = os.path.join(data_dir, scene_name)

            # === Capation Stat
            if mode == "caption":
                caption_path = os.path.join(src_dir, "prompt.json")
                if os.path.exists(caption_path):
                    with open(caption_path, "r") as f:
                        caption = json.load(f)["prompt_camera"]

                    words = caption.strip(".,").split(" ")

                    for word in words:
                        caption_stat_data[word] += 1
                else:
                    log_err(f"{split}/{scene_name} no caption")

            # === Tag Stat
            if mode == "tag":
                cam_tags_path = os.path.join(src_dir, "camera_tags.json")
                if not os.path.exists(cam_tags_path):
                    cam_tags_path = os.path.join(src_dir, "viz", "camera_tags_per_seg.json")

                if not os.path.exists(cam_tags_path):
                    log_err(f"{split}/{scene_name} no cam")
                    continue

                with open(cam_tags_path, "r") as f:
                    tag_data = json.load(f)

                total_len = 0
                static_len = 0
                for vid_seg_idx, data in tag_data.items():
                    cam_seg_str = data["segments"]
                
                    segments = [int(s) for s in cam_seg_str.strip('[]').split(',')]

                    seg_chunks = find_consecutive_chunks(segments)

                    for segment_index, s, e in seg_chunks:
                        s += int(vid_seg_idx) * SEG_LEN
                        e += int(vid_seg_idx) * SEG_LEN
                        chunk_len = e - s
                        total_len += chunk_len

                        cam_index = segment_index//7
                        ang_index = segment_index%7

                        cam_pattern = CAM_INDEX_TO_PATTERN[cam_index]
                        ang_pattern = ANG_INDEX_TO_PATTERN[ang_index]

                        if cam_pattern == "static" and ang_pattern == "static":
                            static_len += chunk_len
                            binary_stat_data["static"] += chunk_len
                        else:
                            binary_stat_data["dynamic"] += chunk_len

                        split_stat_data[split]["cam"][cam_pattern] += chunk_len
                        split_stat_data[split]["ang"][ang_pattern] += chunk_len

                        stat_all_data["cam"][cam_pattern] += chunk_len
                        stat_all_data["ang"][ang_pattern] += chunk_len

                        combined_stat_all_data[f"{cam_pattern} + {ang_pattern}"] += chunk_len
            
            # # scene 기준 80% 이상의 길이에서 static일 경우
            # STATIC_THRES = 0.8
            # if static_len / total_len >= STATIC_THRES:
            #     with open(static_scene_list_path, "a+") as f:
            #         f.write(f"{split}/{scene_name}\n")
            # else:
            #     with open(dynamic_scene_list_path, "a+") as f:
            #         f.write(f"{split}/{scene_name}\n")

    stat_data = {
        "binary": binary_stat_data,
        "all": stat_all_data,
        "split": split_stat_data,
        "caption": caption_stat_data,
        "combined": combined_stat_all_data,
    }
    return stat_data

def merge_nested(d1, d2):
    result = dict(d1)

    for k, v in d2.items():
        if k in result:
            if isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = merge_nested(result[k], v)
            else:
                result[k] += v
        else:
            result[k] = v

    return result

def combine_stats(dict_list):
    return reduce(merge_nested, dict_list)

def launch_stats():
    # static_scene_list_path = f"stats/static_scenes_{postfix}.txt"
    # dynamic_scene_list_path = f"stats/dynamic_scenes_{postfix}.txt"
    # open(static_scene_list_path, "w")
    # open(dynamic_scene_list_path, "w")

    # Ours
    ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
    splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K"]
    dl3dv_stat_data = get_dataset_stat(ROOT_DIR, splits)

    ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
    splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
    dynamicverse_stat_data = get_dataset_stat(ROOT_DIR, splits)

    ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
    splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
    dynpose_stat_data_1 = get_dataset_stat(ROOT_DIR, splits)



    final_stat = combine_stats([dl3dv_stat_data, dynamicverse_stat_data, dynpose_stat_data_1])

    # # DataDoP
    # ROOT_DIR = "/data1/cympyc1785/caption/GenDoP"
    # splits = ["DataDoP_tags"]
    # final_stat = get_dataset_stat(ROOT_DIR, splits)

    # postfix = "dynamicverse"
    # postfix = "all"
    # postfix = "datadop_origin"
    postfix = "ours"

    export_dir = f"analysis/stats/{postfix}"

    with open(f"{export_dir}/stats_{postfix}_split_seg.json", "w+") as f:
        json.dump(final_stat["split"], f, indent=4)
    
    with open(f"{export_dir}/stats_{postfix}_all_seg.json", "w+") as f:
        json.dump(final_stat["all"], f, indent=4)

    with open(f"{export_dir}/stats_{postfix}_all_binary.json", "w") as f:
        json.dump(final_stat["binary"], f, indent=4)
    
    with open(f"{export_dir}/stats_{postfix}_caption.json", "w") as f:
        json.dump(final_stat["caption"], f, indent=4)
    
    with open(f"{export_dir}/stats_{postfix}_combined.json", "w") as f:
        json.dump(final_stat["combined"], f, indent=4)

if __name__ == "__main__":
    launch_stats()