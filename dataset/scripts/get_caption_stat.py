import os
import json
import re
from collections import defaultdict

all_vocab = set()
camera_vocab = set()
prompt_types = [
    "prompt_camera",
    "prompt_dynamic",
    "prompt_scene",
    "prompt_camera_with_scene_video",
    "prompt_camera_with_scene_video_inpainted",
]

def strip_caption(caption):
    caption = caption.strip()
    replace_targets = [".", ":", "\n", "\'", "\"", "\n", "**Concise**", "**Detail**"]
    for target in replace_targets:
        caption = caption.replace(target, "")
    words = caption.split(" ")
    filtered_words = [w.lower() for w in words if re.match(r'^[a-zA-Z]+$', w) and len(w) > 2]
    return filtered_words

def update_captions(ROOT_DIR, splits):
    found = 0
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        for scene_name in sorted(os.listdir(data_dir)):
            caption_path = os.path.join(ROOT_DIR, split, scene_name, "prompts.json")

            if os.path.exists(caption_path) and os.path.getsize(caption_path) > 0:
                found += 1
                with open(caption_path, "r") as f:
                    data = json.load(f)
                for seg_idx_str in data.keys():
                    if "prompt_camera_with_scene_video_inpainted" in data[seg_idx_str].keys():
                        prompt = data[seg_idx_str]["prompt_camera_with_scene_video_inpainted"]["concise"]
                        words = strip_caption(prompt)
                        camera_vocab.update(words)
                    elif "prompt_camera_with_scene_video" in data[seg_idx_str].keys():
                        prompt = data[seg_idx_str]["prompt_camera_with_scene_video"]["concise"]
                        words = strip_caption(prompt)
                        camera_vocab.update(words)
                    
                    for prompt_type in prompt_types:
                        if prompt_type in data[seg_idx_str].keys():
                            prompt = data[seg_idx_str][prompt_type]
                            if isinstance(prompt, dict):
                                concise = prompt["concise"]
                                detail = prompt["detail"]
                                concise_words = strip_caption(concise)
                                detail_words = strip_caption(detail)
                                all_vocab.update(concise_words + detail_words)
                            else:
                                words = strip_caption(prompt)
                                all_vocab.update(words)

    return found

all_words = defaultdict(int)
camera_words = defaultdict(int)
def count_captions(ROOT_DIR, splits):
    found = 0
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        for scene_name in sorted(os.listdir(data_dir)):
            caption_path = os.path.join(ROOT_DIR, split, scene_name, "prompts.json")

            if os.path.exists(caption_path) and os.path.getsize(caption_path) > 0:
                found += 1
                with open(caption_path, "r") as f:
                    data = json.load(f)
                for seg_idx_str in data.keys():
                    if "prompt_camera_with_scene_video_inpainted" in data[seg_idx_str].keys():
                        prompt = data[seg_idx_str]["prompt_camera_with_scene_video_inpainted"]["concise"]
                        words = strip_caption(prompt)
                        for word in words:
                            camera_words[word] += 1
                    elif "prompt_camera_with_scene_video" in data[seg_idx_str].keys():
                        prompt = data[seg_idx_str]["prompt_camera_with_scene_video"]["concise"]
                        words = strip_caption(prompt)
                        for word in words:
                            camera_words[word] += 1
                    
                    for prompt_type in prompt_types:
                        if prompt_type in data[seg_idx_str].keys():
                            prompt = data[seg_idx_str][prompt_type]
                            if isinstance(prompt, dict):
                                concise = prompt["concise"]
                                detail = prompt["detail"]
                                concise_words = strip_caption(concise)
                                detail_words = strip_caption(detail)
                                for word in (concise_words + detail_words):
                                    all_words[word] += 1
                            else:
                                words = strip_caption(prompt)
                                for word in words:
                                    all_words[word] += 1

    return found

ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
# found = update_captions(ROOT_DIR, splits)
found = count_captions(ROOT_DIR, splits)
print(found)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
# found = update_captions(ROOT_DIR, splits)
found = count_captions(ROOT_DIR, splits)
print(found)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
# found = update_captions(ROOT_DIR, splits)
found = count_captions(ROOT_DIR, splits)
print(found)

# print("vocab")
# print("all", len(list(all_vocab)))
# print("cam", len(list(camera_vocab)))

# json.dump(list(all_vocab), open("all_vocab.json", "w"), indent=4)
# json.dump(list(camera_vocab), open("camera_vocab.json", "w"), indent=4)

print("words")
sorted_all_word_dict = dict(sorted(all_words.items(), key=lambda x: x[1], reverse=True))
sorted_cam_word_dict = dict(sorted(camera_words.items(), key=lambda x: x[1], reverse=True))

json.dump(sorted_all_word_dict, open("all_words.json", "w"), indent=4)
json.dump(sorted_cam_word_dict, open("camera_words.json", "w"), indent=4)
