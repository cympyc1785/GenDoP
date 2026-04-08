import os
import json
import subprocess
from tqdm import tqdm
from pathlib import Path
import sys

from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom

dataset_path = '/data1/cympyc1785/SceneData/DL3DV/scenes/7K'
eval_dir = 'eval_results_depth_image'
scene_path_list = [os.path.join(dataset_path, scene_name) for scene_name in sorted(os.listdir(dataset_path))]
batch_size = 64

device = 'cuda:0'
clip_model = load_clip_model('ViT-B/32', device=device)
text_prompt_list = []
scene_name_list = []

# scene_path_list = scene_path_list[:6]
for scene_path in tqdm(scene_path_list):
    scene_name = os.path.basename(scene_path)
    text_path = os.path.join(scene_path, 'prompt.json')
    depth_path = os.path.join(scene_path, 'depths', sorted(os.listdir(os.path.join(scene_path, 'depths')))[0])
    image_path = os.path.join(scene_path, 'images', sorted(os.listdir(os.path.join(scene_path, 'images')))[0])
    with open(text_path, 'r') as f:
        text = json.load(f)['prompt_camera_with_scene_video']['concise']
    if os.path.exists(os.path.join(eval_dir, 'test', f'{scene_name}_caption.json')) is False:
        # cmd = ['python', 'eval.py', 'ArAE', '--workspace', eval_dir,  '--name', f'test/{scene_name}', '--resume', "checkpoints/text_directorial.safetensors", 
        #     '--cond_mode', "text", '--text', text]
        cmd = ['python', 'eval.py', 'ArAE', '--workspace', eval_dir,  '--name', f'test/{scene_name}', '--resume', "checkpoints/text_rgbd.safetensors", 
            '--cond_mode', "depth+image+text", '--text', text, '--image_path', image_path, '--depth_path', depth_path]
        print(cmd)
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout)
            print(out.stderr)
            print("Exit Program")
            sys.exit()
        text_save_path = os.path.join(eval_dir, 'test', f"{scene_name}_caption.json")
        with open(text_save_path, 'w') as f:
            json.dump({
                "Concise Interaction": text
            }, f)
    text_prompt_list.append(text)
    scene_name_list.append(scene_name)

for i in range(0, len(text_prompt_list), batch_size):
    start = i
    end = i + batch_size
    text_prompt = text_prompt_list[start:end]
    scene_name = scene_name_list[start:end]
    seq_embeds, tok_embeds = encode_text(text_prompt, clip_model, max_token_length=None, device=device)
    save_feats_custom(seq_embeds, scene_name, Path(os.path.join(eval_dir, 'seq')))
    save_feats_custom(tok_embeds, scene_name, Path(os.path.join(eval_dir, 'token')))

valid_txt_path = os.path.join(eval_dir, 'test_valid.txt')
with open(valid_txt_path, "w", encoding="utf-8") as f:
    for name in scene_name_list:
        f.write(name + "\n")

    