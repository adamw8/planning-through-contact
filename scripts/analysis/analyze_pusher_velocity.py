import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import zarr


def analyze_dataset(dataset_path):
    root = zarr.open(dataset_path)
    pusher_state = np.array(root["data/state"])
    episode_ends = np.array(root["meta/episode_ends"])

    pusher_speeds = []
    episode_start = 0
    DT = 0.1
    for i, episode_end in enumerate(episode_ends):
        pusher_traj = pusher_state[episode_start:episode_end]

        # Determine the starting index for significant movement
        start_idx = detect_movement_start(pusher_state, threshold=0.001)
        if start_idx < 0:
            continue
        pusher_traj = pusher_traj[start_idx:]

        # Compute average pusher speed
        pusher_speed = np.linalg.norm(np.diff(pusher_traj, axis=0), axis=1).mean() / DT
        pusher_speeds.append(pusher_speed)

        if i > 500:
            break

    print(f"Average pusher speed: {np.mean(pusher_speeds)}")
    print(f"Standard deviation: {np.std(pusher_speeds)}")

    # Plot histogram of pusher speeds
    plt.hist(pusher_speeds, bins=50)
    plt.xlabel("Average Pusher Speed")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pusher Speeds")
    plt.show()


def analyze_eval(combined_pkl_dir):
    pusher_speeds = []
    for i, combined_pkl_file in enumerate(os.listdir(combined_pkl_dir)):
        if not combined_pkl_file.endswith(".pkl"):
            continue
        combined_pkl_path = os.path.join(combined_pkl_dir, combined_pkl_file)
        with open(combined_pkl_path, "rb") as f:
            data = pickle.load(f)
        pusher_data = data.pusher_actual
        t = pusher_data.t
        DT = t[1] - t[0]
        pusher_traj = np.column_stack((pusher_data.x, pusher_data.y))

        # Determine the starting index for significant movement
        start_idx = detect_movement_start(pusher_traj, threshold=0.001)
        if start_idx < 0:
            continue
        pusher_traj = pusher_traj[start_idx:]

        # Compute average pusher speed
        pusher_speed = np.linalg.norm(np.diff(pusher_traj, axis=0), axis=1).mean() / DT
        pusher_speeds.append(pusher_speed)

        if i > 500:
            break

    print(f"Average pusher speed: {np.mean(pusher_speeds)}")
    print(f"Standard deviation: {np.std(pusher_speeds)}")

    # Plot histogram of pusher speeds
    plt.hist(pusher_speeds, bins=50)
    plt.xlabel("Average Pusher Speed")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pusher Speeds")
    plt.show()


def detect_movement_start(pusher_traj, threshold=0.01):
    """
    Detect the index where significant movement starts in the pusher trajectory.
    """
    diffs = np.linalg.norm(np.diff(pusher_traj, axis=0), axis=1)
    start_idx = 0
    for diff in diffs:
        if diff > threshold:
            break
        start_idx += 1

    if start_idx == len(diffs):
        return -1
    return start_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file.")
    args = parser.parse_args()
    if args.dataset_path.endswith(".zarr") or args.dataset_path.endswith(".zarr/"):
        analyze_dataset(args.dataset_path)
    else:
        analyze_eval(args.dataset_path)
