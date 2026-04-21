import os
import numpy as np
import torch
import json
import trimesh
from tqdm import tqdm
from PIL import Image

from data_utils import normalize_camera_extrinsics_and_points

def get_camera_params_from_json(json_path, T=1):
    with open(json_path, 'r') as f:
        data = json.load(f)

    extrinsic_list = []
    intrinsic_list = []
    for params in data[:T]:
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

def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)

def save_pointcloud_datadop(root_dir, depth_root_dir, export_dir):
    for scene_name in tqdm(sorted(os.listdir(root_dir))):
        scene_dir = os.path.join(root_dir, scene_name)
        cameras_path = os.path.join(scene_dir, "cameras.json")

        name_parts = scene_name.split("_")

        origin_scene_name, shot_name = name_parts[0] + "_" + name_parts[1], name_parts[2] + "_" + name_parts[3]

        depth_path = os.path.join(depth_root_dir, origin_scene_name, shot_name + "_depth.npy")
        image_path = os.path.join(depth_root_dir, origin_scene_name, shot_name + "_rgb.png")

        depth = np.load(depth_path)[None, :]

        img = Image.open(image_path).convert("RGB") # force RGB
        img_np = np.array(img, dtype=np.uint8)[None, :]  # (H, W, 3), uint8

        ext_w2c, intrinsic = get_camera_params_from_json(cameras_path)

        pts, cols = _depths_to_world_points_with_colors(
            depth = depth,
            K = intrinsic,
            ext_w2c = ext_w2c,
            images_u8=img_np,
        )

        # pc = trimesh.points.PointCloud(vertices=pts, colors=cols)

        # _, _, avg_scale, _ = normalize_camera_extrinsics_and_points(
        #     extrinsics=torch.from_numpy(ext_w2c),
        #     points=torch.from_numpy(pts),
        # )

        points = torch.from_numpy(np.concatenate([pts, cols.astype(np.float32) / 255.0], axis=1))

        # export_path = os.path.join(export_dir, "points.ply")
        # pc.export(export_path)
        final_export_dir = os.path.join(export_dir, scene_name)
        os.makedirs(final_export_dir, exist_ok=True)
        export_path = os.path.join(final_export_dir, "scene.pt")
        torch.save(points, export_path)


root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/ours"
depth_root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/train"
export_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/ours"
save_pointcloud_datadop(root_dir, depth_root_dir, export_dir)




