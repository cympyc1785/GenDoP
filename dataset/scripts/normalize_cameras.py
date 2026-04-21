import os
import numpy as np
import torch
import json
from tqdm import tqdm

from data_utils import normalize_camera_extrinsics_and_points

def get_camera_params_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    extrinsic_list = []
    intrinsic_list = []
    for params in data:
        R_w2c, t_w2c = params['rotation'], params['position']

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']

        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def save_camera_params(extrinsics, intriniscs, save_path):
    data = []

    assert save_path.endswith(".json")

    extrinsics = np.array(extrinsics)
    intrinsics = np.array(intriniscs)
    assert len(extrinsics) == len(intriniscs)

    for i in range(len(extrinsics)):
        rotation = extrinsics[i][:3, :3].astype(float)
        position = extrinsics[i][:3, 3].astype(float)
        fx = float(intrinsics[i][0, 0])
        fy = float(intrinsics[i][1, 1])
        cx = float(intrinsics[i][0, 2])
        cy = float(intrinsics[i][1, 2])

        data.append({
            "idx": i,
            "rotation": rotation.tolist(),
            "position": position.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        })
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


# root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP"
# splits = ["ours"]
# avg_scale_root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP_avg_scales"
# export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras/DataDoP"

# root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
# avg_scale_root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
# splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
# export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras/DL3DV"

# root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
# avg_scale_root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
# splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
# export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras/DynamicVerse"

root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
avg_scale_root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras/DynamicVerse/dynpose-100k"

for split in splits:
    data_dir = os.path.join(root_dir, split)
    print("processing", split)
    for scene_name in tqdm(sorted(os.listdir(data_dir))):
        scene_dir = os.path.join(data_dir, scene_name)
        cameras_path = os.path.join(scene_dir, "cameras.json")
        prompt_path = os.path.join(scene_dir, "prompts.json")
        avg_scale_dir = os.path.join(avg_scale_root_dir, split, scene_name, "avg_scale")
        if not os.path.exists(prompt_path) or not os.path.getsize(prompt_path) > 0:
            continue
        
        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)

        all_ext_w2c, all_intrinsic = get_camera_params_from_json(cameras_path)

        for seg_idx_str in prompt_data.keys():
            if "frame_idx" not in prompt_data[seg_idx_str].keys():
                continue

            frame_start_idx, frame_end_idx = prompt_data[seg_idx_str]["frame_idx"]
            
            avg_scale_path = os.path.join(avg_scale_dir, f"{int(seg_idx_str)}.json")

            if not os.path.exists(avg_scale_path):
                continue

            avg_scale = json.load(open(avg_scale_path, "r"))

            ext_w2c = all_ext_w2c[frame_start_idx:frame_end_idx]
            intrinsic = all_intrinsic[frame_start_idx:frame_end_idx]

            # Normalize translation
            normalized_extrinsics, _, _, _ = normalize_camera_extrinsics_and_points(torch.tensor(ext_w2c), avg_scale=avg_scale)

            export_path = os.path.join(export_dir, split, scene_name, f"cameras_{int(seg_idx_str)}.json")

            os.makedirs(os.path.dirname(export_path), exist_ok=True)

            save_camera_params(normalized_extrinsics.cpu().numpy(), intrinsic, export_path)



