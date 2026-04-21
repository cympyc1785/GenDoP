import numpy as np
import torch
from core_pkg.common.utils.rotation_utils import compute_rotation_matrix_from_ortho6d

SCENE_CUTTING_THRESHOLD = 90
MAX_POINTS = 16_000_000

def normalize_camera_extrinsics_and_points(extrinsics, points=None, avg_scale=None, scene_cutting_threshold=SCENE_CUTTING_THRESHOLD):
    """
    extrinsics: (N, 4, 4) torch.Tensor (w2c)
    points: (M, 3) torch.Tensor

    normalized_extrinsics: (N, 4, 4)
    normalized_points: (M, 3)
    avg_scale: (1,)
    """
    orig_dtype = extrinsics.dtype
    

    # inverse in float32 for stability (important for bf16)
    e0_inv = torch.linalg.inv(extrinsics[0].float()).to(orig_dtype)

    # normalize extrinsics
    normalized_extrinsics = extrinsics @ e0_inv.unsqueeze(0)

    # normalize points
    R = extrinsics[0, :3, :3]
    t = extrinsics[0, :3, 3]

    normalized_points = None
    mask = None
    
    if points is not None and avg_scale is None:
        points = points.to(dtype=extrinsics.dtype, device=extrinsics.device)
        new_points = points @ R.T + t

        # scale: (1,)
        avg_scale = new_points.norm(dim=-1).mean().unsqueeze(0)

        normalized_points = new_points / avg_scale
        # scene cutting
        d = torch.norm(normalized_points, dim=1)
        # if normalized_points.shape[0] > MAX_POINTS:
        #     idx = torch.randperm(normalized_points.shape[0], device=extrinsics.device)[:MAX_POINTS]
        #     d_sample = d[idx]
        # else:
        #     d_sample = d
        # r = torch.quantile(d_sample, scene_cutting_threshold / 100)
        r = 1.25
        mask = d <= r
    
    normalized_extrinsics[:, :3, 3] /= avg_scale

    return normalized_extrinsics, normalized_points, avg_scale, mask

def out_to_trajectory(out, scale, e0, device=None):
    """
    out: (B, N, 9)
    scale: (B, 1)
    e0: (B, 4, 4)

    matrix_trajectory: (B, N, 4, 4)
    """
    if device is None:
        device = out.device
    if out.shape[-1] > 9:
        out = out[:, :, :9]
    matrix_trajectory = torch.eye(4, device=device)[None].repeat(out.shape[1], 1, 1)[None].repeat(out.shape[0], 1, 1, 1)

    raw_trans = out[:, :, 6:] * scale.unsqueeze(-1)
    matrix_trajectory[:, :, :3, 3] = raw_trans

    rot6d = out[:, :, :6]
    for idx, rot in enumerate(rot6d):
        raw_rot = compute_rotation_matrix_from_ortho6d(rot)
        matrix_trajectory[idx, :, :3, :3] = raw_rot
    matrix_trajectory = matrix_trajectory @ e0.unsqueeze(1)

    return matrix_trajectory

def make_intrinsics(fx_fy, width, height):
    """
    fx_fy:   (B, N, 2)
    width:   (B, N)
    height:  (B, N)

    K: (B, N, 3, 3)
    """
    assert fx_fy.shape[-1] == 2
    assert fx_fy.shape[:2] == width.shape == height.shape

    device = fx_fy.device
    dtype = fx_fy.dtype

    fx = fx_fy[..., 0] * width
    fy = fx_fy[..., 1] * height

    cx = width * 0.5
    cy = height * 0.5

    B, N = width.shape

    K = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)

    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0

    return K

def inverse_camera_matrix(camera_matrix):
    """
    camera_matrix: (..., 4, 4)

    inversed_camera_matrix: (..., 4, 4)
    """
    inversed_camera_matrix = torch.zeros_like(camera_matrix).to(camera_matrix)
    inversed_camera_matrix[..., :3, :3] = camera_matrix[..., :3, :3].transpose(-1, -2)
    inversed_camera_matrix[..., :3, 3:] = -camera_matrix[..., :3, :3].transpose(-1, -2) @ camera_matrix[..., :3, 3:]
    inversed_camera_matrix[..., 3, 3] = 1.0
    
    return inversed_camera_matrix

