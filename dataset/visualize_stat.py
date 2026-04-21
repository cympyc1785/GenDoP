import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.cm as cm


STAT_DIR = "analysis/stats"
VIS_DIR = "analysis/vis"

# postfix = "dynamicverse"
# postfix = "DL3DV"
# postfix = "dynpose-100k"
# postfix = "all"
# postfix = "datadop"
# postfix = "datadop_origin"
postfix = "ours_gendop_style"

def save_fig(keys, values, title, save_name, rotation=90):
    width = max(5, 0.4 * len(keys))
    height = width * 0.5
    fig, ax = plt.subplots(figsize=(width, height))

    values = np.maximum(np.array(values), 1)

    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap("Blues")
    colors = cmap(0.4 + 0.4 * norm(values))

    ax.bar(keys, values, color=colors)

    ax.set_ylabel("Num Frames")
    if title != None:
        ax.set_title(title)
    ax.tick_params(axis='x', rotation=rotation)

    # ax.set_yscale("log")

    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    # ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # ax.yaxis.set_minor_locator(
    #     ticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100)
    # )
    # ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Duration (s)")
    plt.ylabel("Number of Videos")

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/{save_name}", dpi=300)
    plt.close()

    print("saved", f"{VIS_DIR}/{save_name}")

def visualize_combined_stat():
    # Combined All Visualize
    TITLE=f"All Camera Tags Combined (Ours)"
    SAVE_NAME=f"stats_{postfix}_all_seg_combined.png"
    with open(f"{STAT_DIR}/stats_{postfix}_combined.json", "r") as f:
        stat_data = json.load(f)
    sorted_items = sorted(stat_data.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = dict(sorted_items)
    keys = list(sorted_dict.keys())[:40]
    values = list(sorted_dict.values())[:40]
    save_fig(keys, values, TITLE, SAVE_NAME, rotation=90)

def visualize_topk_stat():
    # Top K Visualize
    TITLE=f"{postfix} Camera Tags Top 10"
    SAVE_NAME=f"stats_{postfix}_all_seg_Top10_without_static.png"
    K = 10
    with open(f"{STAT_DIR}/stats_{postfix}_all_seg.json", "r") as f:
        stat_data = json.load(f)
    stat_data_subset = stat_data["cam"]
    stat_data_subset.update(stat_data["ang"])
    stat_data_subset.pop("static")
    sorted_items = sorted(stat_data_subset.items(), key=lambda x: x[1], reverse=True)
    topK = dict(sorted_items[:K])
    others_count = sum(v for _, v in sorted_items[K:])
    topK["Others"] = others_count
    keys = list(topK.keys())
    values = list(topK.values())
    save_fig(keys, values, TITLE, SAVE_NAME)

def visualize_all_tags():
    # Visualize All
    # TITLE=f"{postfix} Camera Tags"
    SAVE_NAME=f"stats_{postfix}_all_seg_cam.png"
    with open(f"{STAT_DIR}/{postfix}/stats_{postfix}_all_seg.json", "r") as f:
        stat_data = json.load(f)
    sorted_items_cam = sorted(stat_data["cam"].items(), key=lambda x: x[1], reverse=True)
    sorted_items_ang = sorted(stat_data["ang"].items(), key=lambda x: x[1], reverse=True)
    cam_dict = dict(sorted_items_cam)
    ang_dict = dict(sorted_items_ang)
    keys = list(cam_dict.keys())
    values = list(cam_dict.values())
    save_fig(keys, values, None, SAVE_NAME)

def visualize_binary_stat():
    # Binary Stat
    TITLE=f"{postfix} Camera Tags Binary"
    SAVE_NAME=f"stats_{postfix}_all_binary.png"
    with open(f"{STAT_DIR}/stats_{postfix}_all_binary.json", "r") as f:
        data = json.load(f)
    keys = data.keys()
    values = data.values()
    save_fig(keys, values, TITLE, SAVE_NAME)

def visualize_binary_separate_stat():
    # Binary Stat Separate Cam & Ang
    other_val_sum = 0
    for key, val in stat_data["ang"].items():
        if key != "static":
            other_val_sum += val
    for key, val in stat_data["cam"].items():
        if key != "static":
            other_val_sum += val
    keys = ["static", "dynamic"]
    values = [stat_data["ang"]["static"] + stat_data["cam"]["static"], other_val_sum]


def visualize_video_length():
    SAVE_NAME=f"shot_length.png"
    with open(f"{STAT_DIR}/ours_video_length/all_image_num_stat.json", "r") as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if int(k) >= 40}
    keys = list(data.keys())
    values = list(data.values())
    save_fig(keys, values, None, SAVE_NAME)

visualize_video_length()
print("Done")

