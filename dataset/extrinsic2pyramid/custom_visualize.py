import os

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
import traceback

from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm

from visualize import quaternion_to_matrix, matrix_to_quaternion, convert_viser_poses_to_new_coordinate_system

def draw_json(json_path, vis_name):
    vis_path = f"{os.path.dirname(json_path)}/{vis_name}.png"
    poses = []
    # data = json.load(open(json_path))['frames']
    data = json.load(open(json_path))
    for i, frame in enumerate(data[:49:5]):
        R_w2c = torch.tensor(frame['rotation'])
        t_w2c = torch.tensor(frame['position'])

        R_c2w = R_w2c.T
        t_c2w = (-R_w2c.T @ t_w2c.unsqueeze(-1)).squeeze(-1)

        c2w = torch.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w

        trans_mat = c2w
        poses.append(trans_mat.tolist())
        
    poses = [poses[i] for i in range(0, len(poses), 2)]
    c2ws = torch.tensor(poses)

    # Convert opencv -> opengl
    c2ws[:, :3, 1:3] *= -1
    
    ref_w2c = torch.inverse(c2ws[:1])
    c2ws = ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1
    
    c2ws = c2ws.numpy()

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

    num_matrices = c2ws.shape[0]

    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{vis_name}/{view['name']}_view.png"
        os.makedirs(vis_name, exist_ok=True)
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

if __name__ == "__main__":
    valid_name_list = []
    vis_name = 'cam_viz_49'
    dataset_dir = "/data1/cympyc1785/caption/GenDoP/SceneData/DL3DV/scenes/1K"
    for name in tqdm(sorted(os.listdir(dataset_dir))):
        json_path = f"{dataset_dir}/{name}/cameras.json"
        vis_path = f"{os.path.dirname(json_path)}/{vis_name}.png"
        if os.path.exists(vis_path):
            print(f"Skip {vis_path}")
            continue
        draw_json(json_path, vis_name)

# from concurrent.futures import ProcessPoolExecutor, as_completed

# visname = "cam_viz_49"
# dataset_dir = "/data1/cympyc1785/caption/GenDoP/SceneData/DL3DV/scenes/1K"

# def worker(name: str):
#     json_file = f"{dataset_dir}/{name}/cameras.json"
#     try:
#         draw_json(json_file, visname)
#     except:
#         print("Failed", name)
#         tb = traceback.format_exc()
#         print("Tracebak:", tb)
#     return ("ok", name)

# if __name__ == "__main__":
#     names = sorted(os.listdir(dataset_dir))

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         futures = [executor.submit(worker, n) for n in names]
#         for _ in tqdm(as_completed(futures), total=len(futures)):
#             pass