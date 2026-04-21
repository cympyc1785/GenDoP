import os
import json
from collections import defaultdict

all_title = defaultdict(int)



def update_captions(ROOT_DIR, splits):
    found = 0
    for split in splits:
        print("processing", split)
        data_dir = os.path.join(ROOT_DIR, split)
        for scene_name in sorted(os.listdir(data_dir)):
            caption_path = os.path.join(ROOT_DIR, split, scene_name, "prompts.json")

            if os.path.exists(caption_path) and os.path.getsize(caption_path) > 0:
                with open(caption_path, "r") as f:
                    data = json.load(f)
                if "0" in data.keys() and "title" in data["0"].keys():
                    title = data["0"]["title"]
                    all_title[title] += 1
                    all_title["total"] += 1
                    found += 1

    return found

total_found = 0
ROOT_DIR = "/data1/cympyc1785/SceneData/DL3DV/scenes"
splits = ["1K", "2K", "3K", "4K", "5K", "6K", "7K",]
found = update_captions(ROOT_DIR, splits)
total_found += found
print(found)

ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes"
splits = ["DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica", "spring", "youtube_vis", "uvo"]
found = update_captions(ROOT_DIR, splits)
total_found += found
print(found)


ROOT_DIR = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k"
splits = [f"dynpose-{i:04d}" for i in range(0, 90)]
found = update_captions(ROOT_DIR, splits)
total_found += found
print(found)

print(total_found)

sorted_dict = dict(sorted(all_title.items(), key=lambda x: x[1], reverse=True))

with open("title_counts.json", "w") as f:
    json.dump(sorted_dict, f, indent=4)

