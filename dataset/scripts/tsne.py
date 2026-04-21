import os
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_traj_vector(traj_path):
    traj = []
    T = 49
    with open(traj_path, "r") as f:
        frames = json.load(f)
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

def get_embedding(candidates, N=1000):
    # random selection
    traj_paths = random.sample(candidates, N)
    vectors = []
    for traj_path in traj_paths:
        traj_vector = load_traj_vector(traj_path)
        vectors.append(traj_vector)

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

emb_ours = get_embedding(candidates_ours)

def run_tsne_camera_trajectories(
    X_dict,                 # {"DL3DV": (N, 12T), "Ours": (N, 12T), ...}
    pca_dim=50,
    tsne_dim=2,
    perplexity=30,
    seed=0,
    standardize=True,
    pca_first=True,
):
    """
    X_dict: dataset_name -> np.ndarray of shape (N, 12*T)
            where each row is one camera trajectory, flattened over time:
            [p1(3), R1(9), p2(3), R2(9), ..., pT(3), RT(9)].

    Returns:
      Z: (sum_N, tsne_dim) t-SNE embedding
      labels: (sum_N,) integer labels per point
      names: list of dataset names aligned to label ids
    """
    names = list(X_dict.keys())
    X_list, labels_list = [], []

    for i, name in enumerate(names):
        X = np.asarray(X_dict[name], dtype=np.float32)
        assert X.ndim == 2, f"{name} must be 2D, got {X.shape}"
        X_list.append(X)
        labels_list.append(np.full((X.shape[0],), i, dtype=np.int32))

    X_all = np.concatenate(X_list, axis=0)              # (N_total, 12T)
    labels = np.concatenate(labels_list, axis=0)        # (N_total,)

    # Optional: standardize features (strongly recommended for t-SNE)
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_all = scaler.fit_transform(X_all)

    # Optional: PCA before t-SNE (strongly recommended)
    if pca_first:
        pca = PCA(n_components=min(pca_dim, X_all.shape[1]), random_state=seed)
        X_all = pca.fit_transform(X_all)                # (N_total, pca_dim)

    # t-SNE
    tsne = TSNE(
        n_components=tsne_dim,
        perplexity=perplexity,
        init="pca" if pca_first else "random",
        learning_rate="auto",
        random_state=seed,
        max_iter=2000,
        verbose=1,
    )
    Z = tsne.fit_transform(X_all)                       # (N_total, tsne_dim)
    return Z, labels, names


def plot_tsne(Z, labels, names, title="Camera Trajectory t-SNE"):
    VIS_DIR = "analysis/vis"
    save_name = "t-sne.png"
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        idx = (labels == i)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.7, label=f"{name} (n={idx.sum()})")
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(markerscale=2, frameon=True)
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/{save_name}", dpi=300)
    plt.close()

    print("saved", f"{VIS_DIR}/{save_name}")


# -------------------------
# Example usage
# -------------------------
X_dict = {
    # "DL3DV": dl3dv_X,    # shape (N1, 12*T)
    "Ours":  emb_ours,     # shape (N2, 12*T)
    # "DynVerse": dyn_X,   # shape (N3, 12*T)
}
Z, labels, names = run_tsne_camera_trajectories(
    X_dict,
    pca_dim=50,
    perplexity=30,
    seed=0,
    standardize=True,
    pca_first=True,
)
plot_tsne(Z, labels, names, title="Camera Trajectory Diversity (t-SNE)")