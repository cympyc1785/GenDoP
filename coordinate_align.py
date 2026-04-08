import json
import torch
import torch.nn.functional as F

def load_cameras(json_path):
    data = json.load(open(json_path))['frames']
    poses = []
    for frame in data:
        poses.append(frame['transform_matrix'])
        
    poses = [poses[i] for i in range(0, len(poses), 2)]
    c2ws = torch.tensor(poses)

    return c2ws



def classify_camera_convention_c2w(c2ws: torch.Tensor):
    """
    Heuristic classifier for camera convention from c2w rotations.
    Returns a dict with scores for likely conventions.
    """
    if c2ws.shape[-2:] == (4, 4):
        R = c2ws[..., :3, :3]
    elif c2ws.shape[-2:] == (3, 4):
        R = c2ws[..., :3, :3]
    elif c2ws.shape[-2:] == (3, 3):
        R = c2ws
    else:
        raise ValueError(f"Unexpected shape {c2ws.shape}")

    # camera axes in world (column vectors)
    xw = R[..., :, 0]  # right
    yw = R[..., :, 1]  # camera +Y axis (down in OpenCV)
    zw = R[..., :, 2]  # camera +Z axis (forward in OpenCV)

    # normalize
    xw = F.normalize(xw, dim=-1)
    yw = F.normalize(yw, dim=-1)
    zw = F.normalize(zw, dim=-1)

    # world canonical axes
    ex = torch.tensor([1., 0., 0.], device=R.device)
    ey = torch.tensor([0., 1., 0.], device=R.device)
    ez = torch.tensor([0., 0., 1.], device=R.device)

    # "Up" in OpenCV is typically -yw, in OpenGL-ish is +yw
    up_from_opencv = -yw
    up_from_opengl =  yw

    # forward in OpenCV is +zw, in OpenGL-ish often -zw
    fwd_from_opencv =  zw
    fwd_from_opengl = -zw

    # scores: how well "up" aligns with +Y (ey) and "forward" aligns with -Z or +Z (depends)
    # For viser, up≈+Y is common. We'll score that.
    up_cv_score = (up_from_opencv * ey).sum(dim=-1).mean().item()
    up_gl_score = (up_from_opengl * ey).sum(dim=-1).mean().item()

    # forward: many viewers use -Z as forward in world view; score alignment to -Z
    neg_ez = -ez
    fwd_cv_score = (fwd_from_opencv * neg_ez).sum(dim=-1).mean().item()
    fwd_gl_score = (fwd_from_opengl * neg_ez).sum(dim=-1).mean().item()

    return {
        "up_align_+Y__assuming_OpenCV(up=-Ycol)": up_cv_score,
        "up_align_+Y__assuming_OpenGL(up=+Ycol)": up_gl_score,
        "fwd_align_-Z__assuming_OpenCV(fwd=+Zcol)": fwd_cv_score,
        "fwd_align_-Z__assuming_OpenGL(fwd=-Zcol)": fwd_gl_score,
        "hint": "Higher is better. If OpenCV-up score is higher, treat camera +Y as down (use -R[:,1] for up).",
    }

c2ws = load_cameras("/data1/cympyc1785/camera_gen/Director3D/exps/tmp2/test/1K_00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24_6_transforms_pred.json")

result = classify_camera_convention_c2w(c2ws)

print(result)