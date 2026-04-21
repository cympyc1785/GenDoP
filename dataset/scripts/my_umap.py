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

def load_traj_vector(traj_path):
    traj = []
    T = 49
    with open(traj_path, "r") as f:
        frames = json.load(f)
    if len(frames) < T:
        return None
    for frame in frames[:T]:
        R_w2c = torch.tensor(frame['rotation'])
        t_w2c = torch.tensor(frame['position'])
        
        R_c2w = R_w2c.T
        t_c2w = (-R_w2c.T @ t_w2c.unsqueeze(-1)).squeeze(-1)

        c2w = torch.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w

        c2w[:3, 1:3] *= -1

        traj.append(c2w.tolist())
    traj = torch.tensor(traj)
    # Normalize
    T0_inv = torch.inverse(traj[0])
    traj = T0_inv.unsqueeze(0) @ traj

    traj = traj[:, :3] # (T, 3, 4)
    traj_vector = traj.reshape(-1) # (12T)
    
    return traj_vector

def get_embedding(candidates, N=5000, max_tries=200000):
    vectors = []
    tries = 0

    while len(vectors) < N and tries < max_tries:
        traj_path = random.choice(candidates)
        v = load_traj_vector(traj_path)
        tries += 1
        if v is None:
            continue
        vectors.append(v)

    if len(vectors) == 0:
        raise RuntimeError("No valid trajectories found (all too short?).")

    if len(vectors) < N:
        print(f"[warn] requested N={N}, but only collected {len(vectors)} valid trajectories.")

    return torch.stack(vectors, dim=0)

def get_candidates(ROOT_DIR, splits):
    candidates = []
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        for scene_name in sorted(os.listdir(data_dir)):
            traj_path = os.path.join(ROOT_DIR, split, scene_name, "cameras.json")
            candidates.append(traj_path)
    return candidates

def run_umap_camera_trajectories(
    X_dict,                 # {"DL3DV": (N, 12T), "Ours": (N, 12T), ...}
    pca_dim=50,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    seed=0,
    standardize=True,
    pca_first=True,
):
    names = list(X_dict.keys())
    X_list, labels_list = [], []

    for i, name in enumerate(names):
        X = X_dict[name].detach().cpu().numpy()
        assert X.ndim == 2, f"{name} must be 2D, got {X.shape}"
        X_list.append(X)
        labels_list.append(np.full((X.shape[0],), i, dtype=np.int32))

    X_all = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if standardize:
        X_all = StandardScaler().fit_transform(X_all)

    if pca_first:
        X_all = PCA(n_components=min(pca_dim, X_all.shape[1]), random_state=seed).fit_transform(X_all)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    Z = reducer.fit_transform(X_all)  # (N_total, 2)
    return Z, labels, names

def plot_umap(Z, labels, names, title="Camera Trajectory UMAP"):
    VIS_DIR = "analysis/vis"
    save_name = "umap.png"
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        idx = labels == i
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.7, label=f"{name} (n={idx.sum()})")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, frameon=True)
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/{save_name}", dpi=300)
    plt.close()

    print("saved", f"{VIS_DIR}/{save_name}")

start_time =time.time()
candidates_ours = []
ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
candidates_ours += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
candidates_ours += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(0, 50)]
candidates_ours += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/data/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(73, 90)]
candidates_ours += get_candidates(ROOT_DIR, splits)

ROOT_DIR = "/data1/cympyc1785/caption/GenDoP/DataDoP"
splits = ["ours"]
candidates_ours += get_candidates(ROOT_DIR, splits)

our_emb = get_embedding(candidates_ours, 10000)

X_dict = {"ours":our_emb}
Z, labels, names = run_umap_camera_trajectories(X_dict)
plot_umap(Z, labels, names)

print("It Took", time.time()-start_time)