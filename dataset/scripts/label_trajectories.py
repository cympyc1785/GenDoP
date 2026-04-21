import os
import json
import shutil

from processing.segmentation import find_consecutive_chunks, ANG_INDEX_TO_PATTERN, CAM_INDEX_TO_PATTERN
from collections import defaultdict


SEG_LEN = 49

def label_trajectories(root_dir, splits, mode="tag"):
    for split in splits:
        if "dynpose-100k" in split:
            continue
        print("split", split)
        data_dir = f"{root_dir}/{split}"
        if not os.path.exists(data_dir):
            continue
        for scene_name in sorted(os.listdir(data_dir)):
            if scene_name == "backup":
                continue
            src_dir = os.path.join(data_dir, scene_name)

            # # === Capation Stat
            # if mode == "caption":
            #     caption_path = os.path.join(src_dir, "prompt.json")
            #     if os.path.exists(caption_path):
            #         with open(caption_path, "r") as f:
            #             caption = json.load(f)["prompt_camera"]

            #         words = caption.strip(".,").split(" ")

            #         for word in words:
            #             caption_stat_data[word] += 1
            #     else:
            #         log_err(f"{split}/{scene_name} no caption")

            # === Tag Stat
            if mode == "tag":
                cam_tags_path = os.path.join(src_dir, "camera_tags.json")
                if not os.path.exists(cam_tags_path):
                    cam_tags_path = os.path.join(src_dir, "viz", "camera_tags_per_seg.json")

                if not os.path.exists(cam_tags_path):
                    print(f"{split}/{scene_name} no cam")
                    continue

                with open(cam_tags_path, "r") as f:
                    tag_data = json.load(f)

                for vid_seg_idx, data in tag_data.items():
                    cam_seg_str = data["segments"]
                
                    segments = [int(s) for s in cam_seg_str.strip('[]').split(',')]

                    seg_chunks = find_consecutive_chunks(segments)

                    stats = {"cam":defaultdict(int),
                        "ang":defaultdict(int),
                        "cam+ang":defaultdict(int)}

                    for segment_index, s, e in seg_chunks:
                        s += int(vid_seg_idx) * SEG_LEN
                        e += int(vid_seg_idx) * SEG_LEN
                        chunk_len = e - s

                        cam_index = segment_index//7
                        ang_index = segment_index%7

                        stats["cam"][cam_index] += chunk_len
                        stats["ang"][ang_index] += chunk_len
                        stats["cam+ang"][segment_index] += chunk_len
                    
                    top1_cam_idx = max(stats["cam"], key=stats["cam"].get)
                    top1_ang_idx = max(stats["ang"], key=stats["ang"].get)
                    top1_cam_ang_idx = max(stats["cam+ang"], key=stats["cam+ang"].get)

                    cam_pattern = CAM_INDEX_TO_PATTERN[top1_cam_idx]
                    ang_pattern = ANG_INDEX_TO_PATTERN[top1_ang_idx]

                    both_cam_pattern = CAM_INDEX_TO_PATTERN[top1_cam_ang_idx//7]
                    both_ang_pattern = ANG_INDEX_TO_PATTERN[top1_cam_ang_idx%7]

                    tag_data[vid_seg_idx]["label"] = {
                        "translation": cam_pattern,
                        "rotation": ang_pattern,
                        "trans+rot": f"{both_cam_pattern} + {both_ang_pattern}"
                    }
                    tag_data[vid_seg_idx]["label_pattern_idx"] ={
                        "translation": top1_cam_idx,
                        "rotation": top1_ang_idx,
                        "trans+rot": top1_cam_ang_idx
                    }
                
                with open(cam_tags_path, "w") as f:
                    json.dump(tag_data, f, indent=4)

# # Ours
# ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
# splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K"]
# dl3dv_stat_data = label_trajectories(ROOT_DIR, splits)

# ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
# splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
# dynamicverse_stat_data = label_trajectories(ROOT_DIR, splits)

# ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
# splits = [f"dynpose-{i:04d}" for i in range(0, 50)]
# dynpose_stat_data_1 = label_trajectories(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(00, 90)]
dynpose_stat_data_1 = label_trajectories(ROOT_DIR, splits)
        

        
