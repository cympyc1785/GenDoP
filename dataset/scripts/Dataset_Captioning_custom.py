import os
import base64
import random
from openai import OpenAI
import json
import numpy as np
from time import sleep
from tqdm import tqdm

def call_local_model(user_prompt, model_name, max_tokens=700, verbose=False):
    client = OpenAI(
        base_url="http://127.0.0.1:22002/v1",
        api_key="none"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        # stream=True,
    )

    return response

def show_content(response):
    print(response.choices[0].message.content)

def save_content(response, file):
    with open(file, 'w') as f:
        f.write(response.choices[0].message.content)

def single_test(prompt_text, caption, video_path):
    prompt_text = prompt_text + "\n\nMovement: " + caption
    print(prompt_text)

    for k in range(3):
        try:
            response = call_gpt4_v(prompt_text, image_path)
            para_len = response.choices[0].message.content.split("\n")
            assert len(para_len) == 3, f"\nLen Error: {para_len}, {response.choices[0].message.content}\n"
            break
        except Exception as e:
            print("Error: ", e)
            continue
    
    save_content(response, relationship_path)

if __name__ == "__main__":
    prompt_path = "./scripts/configs/captioning/llm/relationship+image.json"
    prompts = json.load(open(prompt_path, 'r'))
    prompt_text = prompts['context'] + prompts['instruction'] + prompts['constraint'] + prompts['format']

    VID_SEG_LEN = 49
    ERR_PATH = "errors.txt"
    ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"

    # Add all tasks
    tasks = []
    splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
    for split in splits:
        data_dir = f"{ROOT_DIR}/{split}"
        scene_names = sorted(os.listdir(data_dir))
        for scene_name in scene_names:
            tasks.append((split, scene_name))
    
    # Process Tasks
    for split, scene_name in tqdm(tasks):
        scene_dir = f"{ROOT_DIR}/{split}/{scene_name}"
        tag_path = os.path.join(scene_dir, "prompts.json")

        with open(tag_path, "r") as f:
            all_captions = json.load(f)

        for vid_seg_idx, seg_captions in all_captions.items():
            camera_caption = seg_captions["prompt_camera"]

            single_test(prompt_text, caption, video_path)