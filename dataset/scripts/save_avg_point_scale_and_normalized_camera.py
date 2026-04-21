import os
import numpy as np
import torch
import json
import trimesh

from tqdm import tqdm
from PIL import Image

from data_utils import normalize_camera_extrinsics_and_points
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_monst3r_cam_params(camera_path):
    with open(camera_path, 'r') as f:
        cameras = json.load(f)
    fx = cameras['fl_x']
    fy = cameras['fl_y']
    cx = cameras['cx']
    cy = cameras['cy']
    frames = cameras['frames']
    N = len(frames)

    c2w = np.array([data['transform_matrix'] for data in frames])
    c2w[:, :3, 1:3] *= -1

    R = c2w[:, :3, :3]      # (N,3,3)
    t = c2w[:, :3, 3]       # (N,3)

    R_inv = np.transpose(R, (0, 2, 1))  # R^T
    t_inv = -np.einsum('nij,nj->ni', R_inv, t)

    w2c = np.tile(np.eye(4), (c2w.shape[0], 1, 1))
    w2c[:, :3, :3] = R_inv
    w2c[:, :3, 3] = t_inv
    # e0_inv = np.linalg.inv(w2c[0])
    # normalized_w2c = w2c @ e0_inv[None]

    intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    return w2c, intrinsic

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

def save_camera_params(extrinsics, intrinsics, save_path):
    data = []

    assert save_path.endswith(".json")

    extrinsics = np.array(extrinsics)
    intrinsics = np.array(intrinsics)
    # assert len(extrinsics) == len(intrinsics)
    if intrinsics.ndim == 2:
        for i in range(len(extrinsics)):
            rotation = extrinsics[i][:3, :3].astype(float)
            position = extrinsics[i][:3, 3].astype(float)
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])

            data.append({
                "idx": i,
                "rotation": rotation.tolist(),
                "position": position.tolist(),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            })
    else:
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

def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
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

    pts_all = []

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

        pts_all.append(Xw)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0)

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x

def depths_to_world_points_with_colors_torch(
    depth,          # (N,H,W)  torch.Tensor (float32/float16 OK)
    K,              # (N,3,3)  torch.Tensor
    ext_w2c,         # (N,4,4) or (N,3,4) torch.Tensor
    *,
    device="cuda",
    stride=1,                 # >1이면 다운샘플링
    max_points_per_frame=None,  # int or None (랜덤 샘플링)
    chunk_size=1_000_000,     # 너무 큰 M일 때 matmul을 chunk로
    return_torch=False,
):
    # ---- move / normalize ----
    depth = to_torch(depth).to(device)
    K = to_torch(K).to(device)
    ext_w2c = to_torch(ext_w2c).to(device)

    if depth.dim() != 3:
        raise ValueError(f"depth must be (N,H,W), got {tuple(depth.shape)}")
    N, H, W = depth.shape

    # ext_w2c -> (N,4,4)
    if ext_w2c.shape[-2:] == (3, 4):
        pad = torch.tensor([0, 0, 0, 1], device=device, dtype=ext_w2c.dtype).view(1, 1, 4).repeat(N, 1, 1)
        ext_w2c = torch.cat([ext_w2c, pad], dim=1)
    elif ext_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"ext_w2c must be (N,4,4) or (N,3,4), got {tuple(ext_w2c.shape)}")

    # build pixel grid (downsample by stride)
    us = torch.arange(0, W, stride, device=device)
    vs = torch.arange(0, H, stride, device=device)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")  # (h', w')
    ones = torch.ones_like(uu, dtype=torch.float32)

    # pix: (P,3) float32
    pix = torch.stack([uu, vv, ones], dim=-1).reshape(-1, 3).to(torch.float32)  # (P,3)
    P = pix.shape[0]

    # flatten indices mapping from downsample grid to original image
    # idx in original H*W space for color/depth gather
    flat_idx = (vv * W + uu).reshape(-1)  # (P,)

    pts_list = []

    # precompute K_inv, c2w per frame
    # (safe: torch.linalg.inv on GPU)
    K_inv = torch.linalg.inv(K.to(torch.float32))  # (N,3,3)
    c2w = torch.linalg.inv(ext_w2c.to(torch.float32))  # (N,4,4)

    for i in range(N):
        d = depth[i].reshape(-1)  # (H*W,)
        # gather only downsampled pixels
        d_ds = d.index_select(0, flat_idx)  # (P,)
        valid = torch.isfinite(d_ds) & (d_ds > 0)

        vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # (M,)
        if vidx.numel() == 0:
            continue

        if max_points_per_frame is not None and vidx.numel() > max_points_per_frame:
            perm = torch.randperm(vidx.numel(), device=device)[:max_points_per_frame]
            vidx = vidx.index_select(0, perm)

        # rays = K_inv @ pix^T  but only for vidx
        pix_sel = pix.index_select(0, vidx)  # (M,3)

        # chunked compute to avoid huge intermediate
        M = pix_sel.shape[0]
        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            pix_chunk = pix_sel[start:end]  # (m,3)
            d_chunk = d_ds.index_select(0, vidx[start:end]).to(torch.float32)  # (m,)

            rays = (K_inv[i] @ pix_chunk.t()).t()  # (m,3)
            Xc = rays * d_chunk.unsqueeze(1)       # (m,3)

            Xc_h = torch.cat([Xc, torch.ones((Xc.shape[0], 1), device=device, dtype=torch.float32)], dim=1)  # (m,4)
            Xw = (c2w[i] @ Xc_h.t()).t()[:, :3].contiguous()  # (m,3)

            # colors (uint8)
            pts_list.append(Xw.to(torch.float32))

            start = end

    if len(pts_list) == 0:
        pts = torch.zeros((0, 3), device=device, dtype=torch.float32)
    else:
        pts = torch.cat(pts_list, dim=0)

    if return_torch:
        return pts
    return pts.detach().cpu().numpy()

def save_avg_scale_datadop(root_dir, depth_root_dir, export_dir):
    for scene_name in tqdm(sorted(os.listdir(root_dir))):
        scene_dir = os.path.join(root_dir, scene_name)
        cameras_path = os.path.join(scene_dir, "cameras.json")

        name_parts = scene_name.split("_")

        origin_scene_name, shot_name = name_parts[0] + "_" + name_parts[1], name_parts[2] + "_" + name_parts[3]

        depth_path = os.path.join(depth_root_dir, origin_scene_name, shot_name + "_depth.npy")
        
        image_path = os.path.join(depth_root_dir, origin_scene_name, shot_name + "_rgb.png")

        depth = np.load(depth_path)[None, :]
        image = np.array(Image.open("image.png"), dtype=np.uint8)

        ext_w2c, intrinsic = get_camera_params_from_json(cameras_path)

        pts = depths_to_world_points_with_colors_torch(
            depth = depth,
            K = intrinsic,
            ext_w2c = ext_w2c
        )

        # _, _, avg_scale, _ = normalize_camera_extrinsics_and_points(
        #     extrinsics=torch.from_numpy(ext_w2c),
        #     points=torch.from_numpy(pts),
        # )

        # export_path = os.path.join(export_dir, scene_name, "avg_scale", "0.json")

        # os.makedirs(os.path.dirname(export_path), exist_ok=True)
        # with open(export_path, "w") as f:
        #     json.dump(avg_scale.item(), f)

def save_avg_scale_and_cam_from_point(job):
    split, scene_name, scene_dir, scale_export_dir, cam_export_dir = job

    cameras_path = os.path.join(scene_dir, "cameras.json")

    ext_w2c, intrinsic = get_camera_params_from_json(cameras_path)

    point_path = os.path.join(scene_dir, "scene.pt")
    if not os.path.exists(point_path):
        return
    pts = torch.load(point_path)[:, :3]

    normalized_extrinsics, _, avg_scale, _ = normalize_camera_extrinsics_and_points(
        extrinsics=torch.from_numpy(ext_w2c),
        points=pts,
    )

    scale_export_path = os.path.join(scale_export_dir, split, scene_name, "avg_scale", "0.json")
    cam_export_path = os.path.join(cam_export_dir, split, scene_name, "cameras.json")

    os.makedirs(os.path.dirname(scale_export_path), exist_ok=True)
    with open(scale_export_path, "w") as f:
        json.dump(avg_scale.item(), f)

    os.makedirs(os.path.dirname(cam_export_path), exist_ok=True)
    save_camera_params(normalized_extrinsics.cpu().numpy(), intrinsic, cam_export_path)

def save_avg_scale_and_cam(job):
    split, scene_name, scene_dir, scale_export_dir, cam_export_dir = job

    cameras_path = os.path.join(scene_dir, "cameras.json")
    depth_path = os.path.join(scene_dir, "depths", "000.npy")

    if not os.path.exists(depth_path):
        depth_path = os.path.join(scene_dir, "da3/inpainted/exports/npz/results.npz")
    if not os.path.exists(depth_path):
        depth_path = os.path.join(scene_dir, "da3/inpainted/exports/mini_npz/results.npz")
    if not os.path.exists(depth_path):
        print(depth_path)
        return
    
    try:
        depth = np.load(depth_path)["depth"]
    except:
        return
    if depth.ndim == 2:
        depth = depth[None, :]
    elif depth.ndim == 3:
        depth = depth[:1]
    else:
        return

    ext_w2c, intrinsic = get_camera_params_from_json(cameras_path)

    pts = depths_to_world_points_with_colors_torch(
        depth = depth,
        K = intrinsic[:1],
        ext_w2c = ext_w2c[:1]
    )

    

    # normalized_extrinsics, _, avg_scale, _ = normalize_camera_extrinsics_and_points(
    #     extrinsics=torch.from_numpy(ext_w2c),
    #     points=torch.from_numpy(pts),
    # )

    # scale_export_path = os.path.join(scale_export_dir, split, scene_name, "avg_scale", "0.json")
    # cam_export_path = os.path.join(cam_export_dir, split, scene_name, "cameras.json")

    # os.makedirs(os.path.dirname(scale_export_path), exist_ok=True)
    # with open(scale_export_path, "w") as f:
    #     json.dump(avg_scale.item(), f)

    # os.makedirs(os.path.dirname(cam_export_path), exist_ok=True)
    # save_camera_params(normalized_extrinsics.cpu().numpy(), intrinsic, cam_export_path)


def scale_datadop(root_dir, avg_scale_root_dir, export_dir, data_list_txt):
    with open(data_list_txt, 'r') as f:
        data_list = f.readlines()

    for data_name in tqdm(sorted(data_list)):
        data_name = data_name.replace('\n', '')
        data_path = os.path.join(root_dir, data_name)
        target_dir_path = os.path.join(export_dir, data_name.replace('/', '_'))
        os.makedirs(target_dir_path, exist_ok=True)
        camera_path = data_path + '_transforms_cleaning.json'
        avg_scale_path = os.path.join(avg_scale_root_dir, data_name.replace('/', '_'), "avg_scale", "0.json")

        scale_datadop(camera_path, avg_scale_path, target_dir_path)

        ext_w2c, intrinsic = get_monst3r_cam_params(camera_path)

        if not os.path.exists(avg_scale_path):
            return
        
        with open(avg_scale_path, "r") as f:
            avg_scale = json.load(f)

        normalized_extrinsics, _, _, _ = normalize_camera_extrinsics_and_points(
            extrinsics=torch.from_numpy(ext_w2c),
            avg_scale=avg_scale,
        )

        target_camera_path = os.path.join(target_dir_path, 'cameras.json')

        save_camera_params(normalized_extrinsics, intrinsic, target_camera_path)

def save_avg_scale_from_ply(job):
    split, scene_name, scene_dir, scale_export_dir = job

    cameras_path = os.path.join(scene_dir, "cameras.json")

    ext_w2c, intrinsic = get_camera_params_from_json(cameras_path)

    ply_path = os.path.join(scene_dir, "scene.ply")
    if not os.path.exists(ply_path):
        print("no ply", ply_path)
        return
    
    # pts = torch.load(point_path)[:, :3]
    g = trimesh.load(ply_path, process=False)
    pts = torch.from_numpy(np.asarray(g.vertices, dtype=np.float32))

    for i in range(len(ext_w2c) // 49):
        s = i * 49
        e = (i + 1) * 49
        _, _, avg_scale, _ = normalize_camera_extrinsics_and_points(
            extrinsics=torch.from_numpy(ext_w2c[s:e]),
            points=pts,
        )

        scale_export_path = os.path.join(scale_export_dir, split, scene_name, "avg_scale", f"{i:03d}.json")

        os.makedirs(os.path.dirname(scale_export_path), exist_ok=True)
        with open(scale_export_path, "w") as f:
            json.dump(avg_scale.item(), f)

def parallel_process(worker, jobs, max_workers=4):
    print("Jobs Registered:", len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()


# root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/ours"
# depth_root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP/train"
# export_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP_avg_scales"
# save_avg_scale_datadop(root_dir, depth_root_dir, export_dir)

# root_dir = "/data1/cympyc1785/SceneData/DL3DV/scenes"
# splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
# scale_export_dir = "/data1/cympyc1785/caption/GenDoP/OurDataset_avg_scales/DL3DV"
# cam_export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras_2/DL3DV"
# save_avg_scale_and_cam(root_dir, splits, scale_export_dir, cam_export_dir)

# root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
# splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
# scale_export_dir = "/data1/cympyc1785/caption/GenDoP/OurDataset_avg_scales/DynamicVerse"
# cam_export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras_2/DynamicVerse"
# save_avg_scale_and_cam(root_dir, splits, scale_export_dir, cam_export_dir)

# root_dir = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
# splits = [f"dynpose-{i:04d}" for i in range(66, 90)]
# scale_export_dir = "/data1/cympyc1785/caption/GenDoP/OurDataset_avg_scales/DynamicVerse/dynpose-100k"
# cam_export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras_2/DynamicVerse/dynpose-100k"
# # save_avg_scale_and_cam(root_dir, splits, scale_export_dir, cam_export_dir)

root_dir = "/data1/cympyc1785/SceneData/tartanair/scenes"
splits = sorted(os.listdir(root_dir))
scale_export_dir = "/data1/cympyc1785/SceneData/tartanair/scenes"

jobs = []
for split in splits:
    print("processing", split)
    split_dir = os.path.join(root_dir, split)
    for scene_name in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, scene_name)

        jobs.append((split, scene_name, scene_dir, scale_export_dir))

# save_avg_scale_from_ply(jobs[0])
parallel_process(save_avg_scale_from_ply, jobs, max_workers=8)

# parallel_process(save_avg_scale_and_cam_from_point, jobs, max_workers=8)

# data_type = 'train'
# root_dir = f"/data1/cympyc1785/caption/GenDoP/DataDoP/{data_type}"
# avg_scale_root_dir = "/data1/cympyc1785/caption/GenDoP/DataDoP_avg_scales"
# export_dir = "/data1/cympyc1785/caption/GenDoP/Normalized_Cameras_2/DataDoP_full_train"
# data_list_txt = f'/home/ckd248/data/SCVideo/camera_generation/tools/GenDoP/DataDoP/{data_type}_valid.txt'
# scale_datadop(root_dir, avg_scale_root_dir, export_dir, data_list_txt)


