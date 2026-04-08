import os
import cv2
import tyro
import glob
import time
import json
import math
import shutil
import numpy as np
import torch
from PIL import Image
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import argparse
from tqdm import tqdm

from pathlib import Path

import glob

import kiui
import trimesh
from kiui.op import recenter

from core.options import AllConfigs, Options
from core.models import LMM
from core.utils import monkey_patch_transformers
from core.utils import camera_to_token, camera_to_token_single, token_to_camera, quaternion_to_matrix, matrix_to_quaternion, quaternion_slerp, sample_from_two_pose, sample_from_dense_cameras
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


def draw_json(c2ws, vis_path):
    output_dir = os.path.dirname(vis_path)

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])
    num_matrices = c2ws.shape[0]

    # Create a color gradient from red to purple
    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))  # Each image will be 4x12 inches
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{vis_path[:-4]}_{view['name']}_view.png"
        os.makedirs(output_dir, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
    
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")

    # Now delete the individual view images
    for image_path in image_paths:
        os.remove(image_path)
        print(f"Deleted {image_path}")


def pose_normalize(camera_pose, pred_pose_path, camera_out_seq_len=120):
    camera_pose = camera_pose
    transforms_path = pred_pose_path

    f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()
    # Create a dictionary of intrinsic parameters
    transforms_dict = {
        "w": w,
        "h": h,
        "fl_x": f_x,  # Focal length in x direction
        "fl_y": f_y,  # Focal length in y direction
        "cx": c_x,    # Principal point in x
        "cy": c_y,     # Principal point in y
        'frames': []
    }
    traj_tensor = camera_pose[:,:,:12]
    camera_list = []
    for i in range(camera_out_seq_len):
        t = torch.full((1, 1), fill_value=i/camera_out_seq_len)
        camera = sample_from_dense_cameras(traj_tensor, t)
        camera_list.append(camera[0])
    camera_tensor = torch.cat(camera_list, dim=0)  # Concatenate along the batch dimension (dim=0)
    camera_numpy = camera_tensor.clone().cpu().numpy()
    for idx, row in enumerate(camera_numpy):
        RT = row.reshape(3, 4)
        transform_matrix = np.vstack([RT, [0, 0, 0, 1]])
        transform_matrix_list = transform_matrix.tolist()
        frame_data = {
            "transform_matrix": transform_matrix_list,
            "monst3r_im_id": idx + 1  # Assuming colmap_im_id is an index starting from 1
        }
        transforms_dict['frames'].append(frame_data)

    with open(transforms_path, 'w') as f:
        json.dump(transforms_dict, f, indent=4)

def standard_image(rgb_path, target_height=512, target_width=512):
    image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # [H, W, 4]
    image = image[..., [2, 1, 0]]  # BGR to RGB
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [C, H, W]
    height, width = image_tensor.shape[1], image_tensor.shape[2]

    if height > target_height:
        start_y = (height - target_height) // 2
        image_tensor = image_tensor[:, start_y:start_y + target_height, :]
    
    if width > target_width:
        start_x = (width - target_width) // 2
        image_tensor = image_tensor[:, :, start_x:start_x + target_width]

    if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
        padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
        
        top_padding = (target_height - image_tensor.shape[1]) // 2
        bottom_padding = target_height - image_tensor.shape[1] - top_padding
        left_padding = (target_width - image_tensor.shape[2]) // 2
        right_padding = target_width - image_tensor.shape[2] - left_padding

        padded_image[:, top_padding:top_padding + image_tensor.shape[1], left_padding:left_padding + image_tensor.shape[2]] = image_tensor
        image_tensor = padded_image
    return image_tensor

def standard_depth(depth_path, target_height=512, target_width=512):
    depth_image = np.load(depth_path).astype(np.float32)  # [H, W]
    depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()  # [0, 1]
    
    height, width = depth_tensor.shape[1], depth_tensor.shape[2]

    if height > target_height:
        start_y = (height - target_height) // 2
        depth_tensor = depth_tensor[:, start_y:start_y + target_height, :]

    if width > target_width:
        start_x = (width - target_width) // 2
        depth_tensor = depth_tensor[:, :, start_x:start_x + target_width]

    if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
        padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
        
        top_padding = (target_height - depth_tensor.shape[1]) // 2
        bottom_padding = target_height - depth_tensor.shape[1] - top_padding
        left_padding = (target_width - depth_tensor.shape[2]) // 2
        right_padding = target_width - depth_tensor.shape[2] - left_padding

        padded_depth[:, top_padding:top_padding + depth_tensor.shape[1], left_padding:left_padding + depth_tensor.shape[2]] = depth_tensor
        depth_tensor = padded_depth

    return depth_tensor

def preprocess_sample(opt, output_dir, name, text, image_path, depth_path, text_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # # glob이면 첫 파일 선택
    # def pick_one(pat_or_path):
    #     if pat_or_path is None:
    #         return None
    #     if any(ch in pat_or_path for ch in ["*", "?", "["]):
    #         matches = sorted(glob.glob(pat_or_path))
    #         return matches[0] if len(matches) > 0 else None
    #     return pat_or_path

    # text_path = pick_one(text_path)
    # image_path = pick_one(image_path)
    # depth_path = pick_one(depth_path)

    # # text
    # if text is None and text_path is not None:
    #     info = json.load(open(text_path, 'r'))
    #     text_key = opt.text_key if opt.cond_mode == 'text' else 'Concise Interaction'
    #     text = info[text_key]

    # checks
    if opt.cond_mode == 'text':
        assert text is not None
    elif opt.cond_mode == 'image':
        assert image_path is not None
    elif opt.cond_mode == 'image+text':
        assert text is not None and image_path is not None
    elif opt.cond_mode == 'image+depth':
        assert image_path is not None and depth_path is not None
    elif opt.cond_mode == 'depth+image+text':
        assert text is not None and image_path is not None and depth_path is not None
    else:
        raise ValueError(opt.cond_mode)

    rgb = None
    depth = None

    if image_path is not None and opt.cond_mode != 'text':
        rgb = standard_image(image_path, target_height=opt.target_height, target_width=opt.target_width)  # (3,H,W)

    if depth_path is not None and ("depth" in opt.cond_mode):
        depth = standard_depth(depth_path, target_height=opt.target_height, target_width=opt.target_width)  # (1,H,W)

    return {
        "name": name,
        "output_dir": output_dir,
        "text": text,
        "rgb": rgb,
        "depth": depth,
        "text_path": text_path,
        "image_path": image_path,
        "depth_path": depth_path,
    }

class DataDoPDataset(Dataset):
    def __init__(self, opt, root_dir, workspace):
        self.opt = opt
        self.root_dir = root_dir
        self.workspace = workspace
        self.scene_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        scene_dir = os.path.join(self.root_dir, scene_name)

        text_pat  = os.path.join(scene_dir, "*_caption.json")
        rgb_pat   = os.path.join(scene_dir, "*_rgb.png")
        depth_pat = os.path.join(scene_dir, "*_depth.npy")

        out_dir = os.path.join(self.workspace, scene_name)

        sample = preprocess_sample(
            self.opt,
            out_dir,
            scene_name,
            text=None,
            text_path=text_pat,
            image_path=rgb_pat,
            depth_path=depth_pat,
        )
        return sample

class OurDataset(Dataset):
    def __init__(self, opt, tasks, output_dir, workspace):
        self.opt = opt
        self.root_dir = "/data1/cympyc1785/SceneData"
        self.workspace = workspace
        self.tasks = tasks
        self.out_dir = output_dir

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        scene_dir, seg_idx_str = self.tasks[idx]

        base_path = Path(scene_dir)

        data_name = f"{base_path.parents[0].name}_{base_path.name}_{seg_idx_str}"

        if "DL3DV" in scene_dir:
            dataset = "DL3DV"
        elif "dynpose" in scene_dir:
            dataset = "dynpose-100k"
        else:
            dataset = "DynamicVerse"

        prompt_path  = os.path.join(scene_dir, "prompts.json")
        if not os.path.exists(prompt_path):
            print("no text", prompt_path)
            raise ValueError("no text", scene_dir, seg_idx_str)

        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)

        if not "frame_idx" in prompt_data[seg_idx_str].keys():
            print("no frame idx", prompt_path)
            raise ValueError("no frame idx", scene_dir, seg_idx_str)
        
        s, e = prompt_data[seg_idx_str]["frame_idx"]
        
        if "prompt_camera_with_scene_video_inpainted" in prompt_data[seg_idx_str].keys():
            text = prompt_data[seg_idx_str]["prompt_camera_with_scene_video_inpainted"]["concise"]
        elif "prompt_camera_with_scene_video" in prompt_data[seg_idx_str].keys():
            text = prompt_data[seg_idx_str]["prompt_camera_with_scene_video"]["concise"]
        
        if dataset == "DL3DV":
            rgb_dir = os.path.join(scene_dir, "images")
            if not os.path.exists(rgb_dir):
                print("no image", scene_dir)
                raise ValueError("no image", scene_dir, seg_idx_str)
            rgb_path = os.path.join(rgb_dir, sorted(os.listdir(rgb_dir))[s])
        else:
            rgb_dir = os.path.join(scene_dir, "da3_imgs")
            if not os.path.exists(rgb_dir):
                print("no image", scene_dir)
                raise ValueError("no image", scene_dir, seg_idx_str)
            for filename in os.listdir(rgb_dir):
                if int(filename.split(".")[0]) == s:
                    rgb_path = os.path.join(rgb_dir, filename)
                    break
        
        depth_dir = os.path.join(scene_dir, "depths")
        if not os.path.exists(depth_dir):
            print("no depth", scene_dir)
            raise ValueError("no depth", scene_dir, seg_idx_str)
        
        depth_path = None
        for filename in os.listdir(depth_dir):
            if int(filename.split(".")[0]) == s:
                depth_path = os.path.join(depth_dir, filename)
                break
        if depth_path is None:
            print("no depth", scene_dir)
            raise ValueError("no depth", scene_dir, seg_idx_str)

        sample = preprocess_sample(
            self.opt,
            self.out_dir,
            data_name,
            text=text,
            image_path=rgb_path,
            depth_path=depth_path,
        )
        return sample

def collate_to_conds(opt):
    def _collate(batch):
        # batch: list of sample dicts
        names = [b["name"] for b in batch]
        out_dirs = [b["output_dir"] for b in batch]

        texts = [b["text"] for b in batch]
        rgbs = [b["rgb"] for b in batch if b["rgb"] is not None]
        depths = [b["depth"] for b in batch if b["depth"] is not None]

        # build conds exactly like your original code expects
        if opt.cond_mode == "text":
            conds = texts  # List[str] length B
        elif opt.cond_mode == "image":
            rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)  # (B,3,H,W)
            conds = rgb_batch
        elif opt.cond_mode == "image+text":
            rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)
            conds = [texts, rgb_batch]
        elif opt.cond_mode == "image+depth":
            rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)
            depth_batch = torch.stack([b["depth"] for b in batch], dim=0)  # (B,1,H,W)
            conds = [depth_batch, rgb_batch]
        elif opt.cond_mode == "depth+image+text":
            rgb_batch = torch.stack([b["rgb"] for b in batch], dim=0)
            depth_batch = torch.stack([b["depth"] for b in batch], dim=0)
            conds = [texts, rgb_batch, depth_batch]
        else:
            raise ValueError(opt.cond_mode)

        meta = {
            "names": names,
            "output_dirs": out_dirs,
            "texts": texts,
            "text_paths": [b["text_path"] for b in batch],
        }
        return conds, meta

    return _collate

def postprocess_and_save_tokens_batch(opt, tokens, meta):
    """
    tokens: model.generate outputs for batch (len B)
    meta: dict with per-sample lists:
      meta["names"][i]
      meta["output_dirs"][i]
      meta["texts"][i]
      meta["text_paths"][i]
    """

    discrete_bins = opt.discrete_bins
    row_to_add = np.array([0, 0, 0, 1])

    # tokens를 for로 도는 구조 유지
    for i, token in enumerate(tokens):
        name = meta["names"][i]
        output_dir = meta["output_dirs"][i]
        text = meta["texts"][i]
        text_path = meta["text_paths"][i]

        os.makedirs(output_dir, exist_ok=True)

        # ---- token -> coords (원형 유지) ----
        if token[:-1].shape[0] != opt.pose_length * 10:
            token = torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60],
                                 device=token.device) / 256 * discrete_bins
            token = token.repeat(opt.pose_length)
            coords = token.reshape(-1, 10)
        else:
            coords = token[:-1].reshape(-1, 10)

        # 원래 코드처럼 다시 tensor로 감싸는 흐름 유지(사실 불필요하지만 동일하게)
        coords = torch.tensor(coords, dtype=torch.float32, device=token.device)

        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        temp_traj = coords_traj / (0.5 * discrete_bins) - 1
        temp_instri = coords_instri / (discrete_bins / 10)
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2)

        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.expand(1, -1, -1)
        camera_pose = token_to_camera(camera_tokens, 512, 512)

        # ---- c2ws 만들기 (원형 유지) ----
        c2ws = np.array(camera_pose[:, :, :12].cpu())
        scale_value = np.array(scale[0].detach().cpu())
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * scale_value
        c2ws = np.array([np.vstack((matrix, row_to_add)) for matrix in c2ws])

        # ---- traj 이미지 (meta.name -> name으로) ----
        # draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))

        # # ---- GT traj 복사 (원형 유지) ----
        # if text_path is not None and os.path.exists(text_path):
        #     gt_traj_path = text_path.replace("_caption.json", "_traj_cleaning.png")
        #     new_traj_path = os.path.join(output_dir, f"{name}_traj_GT.png")
        #     if os.path.exists(gt_traj_path):
        #         shutil.copy(gt_traj_path, new_traj_path)

        pred_pose_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
        pose_normalize(camera_pose, pred_pose_path, 49)

    torch.cuda.synchronize()

def process_data(opt, tasks, output_dir):
    # root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/test"
    # dataset = DataDoPDataset(opt, root_dir=root_dir, workspace=opt.workspace)
    dataset = OurDataset(opt, tasks=tasks, output_dir=output_dir, workspace=opt.workspace)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_to_conds(opt),
    )

    # Inference
    
    for conds, meta in tqdm(loader):
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tokens = model.generate(conds, max_new_tokens=opt.test_max_seq_length, clean=True)

        t1 = time.time()
        # print(f'[INFO] Processing, time = {t1 - t0:.4f}s')

        postprocess_and_save_tokens_batch(opt, tokens, meta)



if __name__ == "__main__":
    opt = tyro.cli(AllConfigs)

    if opt.cond_mode == 'text':
        opt.num_cond_tokens = 77
    elif opt.cond_mode == 'depth+image+text':
        opt.num_cond_tokens = 591
        
    monkey_patch_transformers()
    kiui.seed_everything(opt.seed)
    # model
    model = LMM(opt)

    # resume pretrained checkpoint
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        print(f'[WARN] model randomly initialized, are you sane?')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.half().eval().to(device)

    output_dir = "outputs/test"
    # image_path = opt.image_path
    # text = opt.text
    # text_path = opt.text_path
    # depth_path = opt.depth_path

    with open("test_data_final.txt", "r") as f:
        task_lines = f.readlines()

    tasks = []
    for line in task_lines:
        target_path = line.strip().replace("\n", "")
        scene_dir = os.path.dirname(target_path)
        seg_idx_str = os.path.basename(target_path)

        tasks.append((scene_dir, seg_idx_str))
    print("Job Registered:", len(tasks))

    process_data(opt, tasks, output_dir)
