import argparse

import cv2
import numpy as np
import zarr


def play_trajectory_videos(zarr_path, fps, stride=1, image_size=None):
    """
    Play trajectory videos from the overhead camera data in the Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        fps (int): Frames per second for the video.
        stride (int): Number of episodes to skip when playing videos.
        image_size (tuple): Desired image size (width, height) for resizing. None for original size.
    """
    # Load Zarr dataset
    dataset = zarr.open(zarr_path, mode="r")

    # Extract overhead camera data and episode ends
    overhead_camera = dataset["data"]["overhead_camera"]
    episode_ends = dataset["meta"]["episode_ends"][:]

    # Compute start indices for each episode
    episode_starts = [0] + episode_ends[:-1].tolist()

    # Save last image for first trajectory
    example_image = overhead_camera[episode_ends[1750] - 1]
    example_image = cv2.cvtColor(example_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("overhead_camera_images/overhead_camera.png", example_image)
    return

    for i in range(0, len(episode_starts), stride):
        start_idx = episode_starts[i]
        end_idx = episode_ends[i]

        # Extract images for the current episode
        trajectory_images = overhead_camera[start_idx:end_idx]

        # Convert images to BGR (for opencv)
        trajectory_images = np.array(
            [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in trajectory_images]
        )

        # Resize images if image_size is specified
        if image_size is not None:
            trajectory_images = np.array(
                [cv2.resize(img, image_size) for img in trajectory_images]
            )

        # Display the video
        for img in trajectory_images:
            cv2.imshow(f"Trajectory {i + 1}", img)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                print("Video playback interrupted by user.")
                return

        # Close the window after the trajectory video ends
        cv2.destroyWindow(f"Trajectory {i + 1}")


if __name__ == "__main__":
    """
    Example usage:
    python scripts/analysis/visualize_dataset.py --zarr-path ~/workspace/gcs-diffusion/data/planar_pushing_cotrain/sim_tee_data.zarr --fps 30 --stride 1 --image-size 640x480
    """
    parser = argparse.ArgumentParser(
        description="Play trajectory videos from a Zarr dataset."
    )
    parser.add_argument(
        "--zarr-path", type=str, required=True, help="Path to the Zarr dataset."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the video."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Number of episodes to skip between videos.",
    )
    parser.add_argument(
        "--image-size",
        type=lambda s: tuple(map(int, s.split("x"))),
        default=None,
        help="Desired image size in the format WIDTHxHEIGHT (e.g., 640x480).",
    )
    args = parser.parse_args()

    play_trajectory_videos(
        zarr_path=args.zarr_path,
        fps=args.fps,
        stride=args.stride,
        image_size=args.image_size,
    )
