import os
import json
import random
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

import torch
import json

from pathlib import Path

from evo.tools.file_interface import read_kitti_poses_file

def get_motion(c2w):
    pass

def load_traj_and_motion(traj_path, T=49):
    traj = []

    with open(traj_path, "r") as f:
        frames = json.load(f)

    # if len(frames) < T:
    #     return None, None

    # -------------------------
    # build c2w trajectory
    # -------------------------
    for frame in frames:
        R_w2c = torch.tensor(frame['rotation'], dtype=torch.float32)
        t_w2c = torch.tensor(frame['position'], dtype=torch.float32)

        R_c2w = R_w2c.T
        t_c2w = (-R_w2c.T @ t_w2c.unsqueeze(-1)).squeeze(-1)

        c2w = torch.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w

        traj.append(c2w)

    traj = torch.stack(traj)  # (T, 4, 4)

    # -------------------------
    # normalize to first frame
    # -------------------------
    T0_inv = torch.inverse(traj[0])
    traj = T0_inv.unsqueeze(0) @ traj  # (T,4,4)

    # =========================
    # Compute absolute cumsum
    # =========================

    R = traj[:, :3, :3]
    t = traj[:, :3, 3]

    # ---- Translation ----
    dt = t[1:] - t[:-1]                   # (T-1,3)
    step_translation = torch.norm(dt, dim=-1)
    total_translation = step_translation.sum()

    # ---- Rotation ----
    R_prev = R[:-1]
    R_next = R[1:]
    R_rel = R_prev.transpose(-1, -2) @ R_next

    trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)          # (T-1,)
    total_rotation = theta.sum()

    return total_translation.item(), total_rotation.item()

def get_vectors(candidates, N=5000, max_tries=200000):
    vectors = []
    tries = 0

    while len(vectors) < N and tries < max_tries:
        traj_path = random.choice(candidates)
        total_translation, total_rotation = load_traj_and_motion(traj_path)
        tries += 1
        if total_translation is None:
            continue
        vectors.append([total_translation, total_rotation])

    if len(vectors) == 0:
        raise RuntimeError("No valid trajectories found (all too short?).")

    if len(vectors) < N:
        print(f"[warn] requested N={N}, but only collected {len(vectors)} valid trajectories.")

    vectors = torch.tensor(vectors)

    filtered = filter_by_quantile(vectors)

    return filtered

def get_candidates(ROOT_DIR, splits):
    candidates = []
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        for scene_name in sorted(os.listdir(data_dir)):
            traj_path = os.path.join(ROOT_DIR, split, scene_name, "cameras.json")
            candidates.append(traj_path)
    return candidates

def filter_by_quantile(vectors, q=0.98):
    """
    vectors: (N, 2)
    q: 상위 q 분위까지만 유지 (ex: 0.98 → 상위 2% 제거)
    """

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu()

    trans = vectors[:, 0]
    rot = vectors[:, 1]

    trans_th = torch.quantile(trans, q)
    rot_th = torch.quantile(rot, q)

    mask = (trans <= trans_th) & (rot <= rot_th)

    filtered = vectors[mask]

    print(f"Original: {len(vectors)}")
    print(f"Filtered: {len(filtered)} (q={q})")

    return filtered

def visualize_motion_embedding(vectors, save_path="motion_scatter.png", title="Camera Motion Distribution", log_scale=False):
    """
    vectors: (N, 2) tensor
    """

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu().numpy()

    x = vectors[:, 0]  # translation
    y = vectors[:, 1]  # rotation (rad)

    # rad → degree (가독성 좋게)
    y = y * 180 / np.pi

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, s=10)

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # plt.xlabel("Total Translation")
    # plt.ylabel("Total Rotation (deg)")
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved to {save_path}")

def visualize_multiple_motion_embeddings(
    dataset_dict,
    save_path="motion_multi_scatter.png",
    title="Camera Motion Distribution",
    log_scale=False,
    alpha=0.5,
    point_size=10,
):
    """
    dataset_dict: {
        "name1": tensor(N1,2),
        "name2": tensor(N2,2),
        ...
    }
    """

    plt.figure(figsize=(7, 7))

    colors = plt.cm.tab10.colors  # 색깔 자동 분배

    for i, (name, vectors) in enumerate(dataset_dict.items()):

        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy()

        x = vectors[:, 0]
        y = vectors[:, 1] * 180 / np.pi  # rad → deg

        color = colors[i % len(colors)]

        plt.scatter(
            x,
            y,
            alpha=alpha,
            s=point_size,
            label=name,
            color=color,
        )

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel("Total Translation (m)")
    plt.ylabel("Total Rotation (deg)")
    # plt.title(title)

    plt.xticks([])
    plt.yticks([])

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved to {save_path}")

start_time =time.time()
ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
dl3dv_candidates = get_candidates(ROOT_DIR, splits)

dynamicverse_candidates = []
ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
dynamicverse_candidates += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(0, 50)]
dynamicverse_candidates += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(50, 90)]
dynamicverse_candidates += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/caption/GenDoP/DataDoP"
splits = ["ours"]
datadop_candidates = get_candidates(ROOT_DIR, splits)

dl3dv_vectors = get_vectors(dl3dv_candidates, 10000)
dl3dv_vectors /= 6
visualize_motion_embedding(dl3dv_vectors, save_path="motion_scatter_dl3dv.png", title="Camera Motion Distribution (DL3DV)")

dynamicverse_vectors = get_vectors(dynamicverse_candidates, 10000)
dynamicverse_vectors /= 3
visualize_motion_embedding(dynamicverse_vectors, save_path="motion_scatter_dynamicverse.png", title="Camera Motion Distribution (DynamicVerse)")

datadop_vectors = get_vectors(datadop_candidates, 10000)
visualize_motion_embedding(datadop_vectors, save_path="motion_scatter_datadop.png", title="Camera Motion Distribution (DataDoP)")

# candidates_ours = dl3dv_candidates + dynamicverse_candidates + datadop_candidates
# ours_vectors = get_vectors(candidates_ours, 10000)
# visualize_motion_embedding(ours_vectors, save_path="motion_scatter_all_dataset.png", title="Camera Motion Distribution (All Dataset)")

# et_candidates = get_et_candidates()
# et_vectors = get_vectors()


dataset_dict = {
    "dl3dv": dl3dv_vectors,
    "dynamicverse": dynamicverse_vectors,
    "datadop": datadop_vectors,
}

visualize_multiple_motion_embeddings(dataset_dict,
    save_path="motion_multi_dataset_scatter_compare.png",
    title="Camera Motion Distribution Compare",)

print("It Took", time.time()-start_time)