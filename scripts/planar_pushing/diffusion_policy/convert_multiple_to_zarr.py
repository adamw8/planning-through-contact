import importlib
import logging
import math
import os
import pathlib
import pickle
import shutil
from typing import List, Optional, Tuple

import cv2
import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from PIL import Image
from pydrake.all import Meshcat, StartMeshcat
from tqdm import tqdm

from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.simulation.controllers.replay_position_source import (
    ReplayPositionSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.data_collection_table_environment import (
    DataCollectionConfig,
    DataCollectionTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    get_slider_pose_within_workspace,
    get_slider_sdf_path,
    models_folder,
)
from planning_through_contact.tools.utils import (
    create_processed_mesh_primitive_sdf_file,
    load_primitive_info,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar_pushing import make_traj_figure


def convert_to_zarr(
    rendered_plans_dirs: List[str],
    zarr_path: str,
    sim_config: PlanarPushingSimConfig,
    data_collection_config: DataCollectionConfig,
    debug: bool = False,
):
    """
    Converts the rendered plans to zarr format.

    This function has 2 modes (regular or reduce).
    Regular mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── 0
    ├──├──images
    ├──├──log.txt
    ├──├──planar_position_command.pkl
    ├── 1
    ...
    In regular mode, this function loops through all trajectories and saves the data to zarr format.

    Reduce mode: Assume the rendered plans trajectory has the following structure

    rendered_plans_dir
    ├── run_0
    ├──├── 0
    ├──├──├──images
    ├──├──├──log.txt
    ├──├──├──planar_position_command.pkl
    ├──├── 1
    ...
    ├── run_1
    ...

    In reduce mode, this function loops through all the runs. Each run contains trajectories.
    This mode is most likely used for MIT Supercloud where data generation is parallelized
    over multiple runs.
    """

    print("\nConverting data to zarr...")

    # Collect all the data paths to compress into zarr format
    traj_dir_list = []
    for rendered_plans_dir in rendered_plans_dirs:
        rendered_plans_dir = pathlib.Path(rendered_plans_dir)
        for plan in os.listdir(rendered_plans_dir):
            traj_dir = rendered_plans_dir.joinpath(plan)
            if not os.path.isdir(traj_dir):
                continue
            traj_dir_list.append(traj_dir)

    concatenated_states = []
    concatenated_slider_states = []
    concatenated_actions = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0

    freq = data_collection_config.policy_freq
    dt = 1 / freq

    num_ik_fails = 0
    num_angular_speed_violations = 0

    for traj_dir in tqdm(traj_dir_list):
        traj_log_path = traj_dir.joinpath("combined_logs.pkl")
        log_path = traj_dir.joinpath("log.txt")

        # If too many IK fails, skip this rollout
        if _is_ik_fail(log_path):
            num_ik_fails += 1
            continue

        # load pickle file and timing variables
        combined_logs = pickle.load(open(traj_log_path, "rb"))
        pusher_desired = combined_logs.pusher_desired
        slider_desired = combined_logs.slider_desired

        if _has_high_angular_speed(
            slider_desired,
            data_collection_config.angular_speed_threshold,
            data_collection_config.angular_speed_window_size,
        ):
            num_angular_speed_violations += 1
            continue

        t = pusher_desired.t
        total_time = math.floor(t[-1] * freq) / freq

        # get start time
        start_idx = _get_start_idx(pusher_desired)
        start_time = math.ceil(t[start_idx] * freq) / freq

        # get state, action, images
        current_time = start_time
        idx = start_idx
        state = []
        slider_state = []

        while current_time < total_time:
            # state and action
            idx = _get_closest_index(t, current_time, idx)
            current_state = np.array(
                [
                    pusher_desired.x[idx],
                    pusher_desired.y[idx],
                    pusher_desired.theta[idx],
                ]
            )
            current_slider_state = np.array(
                [
                    slider_desired.x[idx],
                    slider_desired.y[idx],
                    slider_desired.theta[idx],
                ]
            )
            state.append(current_state)
            slider_state.append(current_slider_state)

            # update current time
            current_time = round((current_time + dt) * freq) / freq

        state = np.array(state)  # T x 3
        slider_state = np.array(slider_state)  # T x 3
        action = np.array(state)[:, :2]  # T x 2
        action = np.concatenate([action[1:, :], action[-1:, :]], axis=0)  # shift action

        # get target
        target = np.array([slider_state[-1] for _ in range(len(state))])

        # update concatenated arrays
        concatenated_states.append(state)
        concatenated_slider_states.append(slider_state)
        concatenated_actions.append(action)
        concatenated_targets.append(target)
        episode_ends.append(current_end + len(state))
        current_end += len(state)

    assert num_ik_fails + num_angular_speed_violations + len(episode_ends) == len(
        traj_dir_list
    )
    print(
        f"{num_ik_fails} of {len(traj_dir_list)} rollouts were skipped due to IK fails."
    )
    print(
        f"{num_angular_speed_violations} of {len(traj_dir_list)} rollouts were skipped due to high angular speeds."
    )
    print(f"Total number of converted rollouts: {len(episode_ends)}\n")

    # save to zarr
    root = zarr.open_group(zarr_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (data_collection_config.state_chunk_length, state.shape[1])
    slider_state_chunk_size = (
        data_collection_config.state_chunk_length,
        state.shape[1],
    )
    action_chunk_size = (data_collection_config.action_chunk_length, action.shape[1])
    target_chunk_size = (data_collection_config.target_chunk_length, target.shape[1])

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_slider_states = np.concatenate(concatenated_slider_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    episode_ends = np.array(episode_ends)
    last_episode_end = episode_ends[-1]

    assert last_episode_end == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_slider_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_targets.shape[0]

    data_group.create_dataset(
        "state", data=concatenated_states, chunks=state_chunk_size
    )
    data_group.create_dataset(
        "slider_state", data=concatenated_slider_states, chunks=slider_state_chunk_size
    )
    data_group.create_dataset(
        "action", data=concatenated_actions, chunks=action_chunk_size
    )
    data_group.create_dataset(
        "target", data=concatenated_targets, chunks=target_chunk_size
    )
    meta_group.create_dataset("episode_ends", data=episode_ends)

    # Delete arrays to save memory
    del concatenated_states
    del concatenated_slider_states
    del concatenated_actions
    del concatenated_targets
    del episode_ends

    # Save images separately and one at a time to save RAM
    camera_names = [camera_config.name for camera_config in sim_config.camera_configs]
    desired_image_shape = np.array(
        [data_collection_config.image_height, data_collection_config.image_width, 3]
    )
    image_chunk_size = [
        data_collection_config.image_chunk_length,
        *desired_image_shape,
    ]

    for camera_name in camera_names:
        print(f"Converting images from {camera_name} to zarr...")
        concatenated_images = zarr.zeros(
            (last_episode_end, *desired_image_shape),
            chunks=image_chunk_size,
            dtype="u1",
        )
        sequence_idx = 0

        for traj_dir in tqdm(traj_dir_list):
            traj_log_path = traj_dir.joinpath("combined_logs.pkl")
            log_path = traj_dir.joinpath("log.txt")

            # If too many IK fails, skip this rollout
            if _is_ik_fail(log_path):
                continue

            # load pickle file and timing variables
            combined_logs = pickle.load(open(traj_log_path, "rb"))
            pusher_desired = combined_logs.pusher_desired
            total_time = pusher_desired.t[-1]
            total_time = math.floor(total_time * freq) / freq

            if _has_high_angular_speed(
                combined_logs.slider_desired,
                data_collection_config.angular_speed_threshold,
                data_collection_config.angular_speed_window_size,
            ):
                continue

            # get start time
            start_idx = _get_start_idx(pusher_desired)
            start_time = math.ceil(t[start_idx] * freq) / freq
            del pusher_desired

            # get state, action, images
            current_time = start_time
            idx = start_idx

            while current_time < total_time:
                idx = _get_closest_index(t, current_time, idx)

                # Image names are "{time in ms}" rounded to the nearest 100th
                image_name = round((current_time * 1000) / 100) * 100
                image_path = traj_dir.joinpath(camera_name, f"{int(image_name)}.png")
                img = Image.open(image_path).convert("RGB")
                img = np.asarray(img)
                if not np.allclose(img.shape, desired_image_shape):
                    # Image size for cv2 is (width, height) instead of (height, width)
                    img = cv2.resize(
                        img, (desired_image_shape[1], desired_image_shape[0])
                    )

                concatenated_images[sequence_idx] = img
                sequence_idx += 1

                if debug:
                    from matplotlib import pyplot as plt

                    print(f"\nCurrent time: {current_time}")
                    print(f"Current index: {idx}")
                    print(f"Image path: {image_path}")
                    plt.imshow(img[6:-6, 6:-6, :])
                    plt.show()

                current_time = round((current_time + dt) * freq) / freq
            # End episode time step loop
        # End episode loop

        # Save images to zarr
        assert len(concatenated_images) == last_episode_end
        assert sequence_idx == last_episode_end
        data_group.create_dataset(
            camera_name,
            data=concatenated_images,
            chunks=image_chunk_size,
        )

    # End camera loop


def _get_start_idx(pusher_desired):
    """
    Finds the index of the first "non-stationary" command.
    This is the index of the start of the trajectory.
    """

    length = len(pusher_desired.t)
    first_non_zero_idx = 0
    for i in range(length):
        if (
            pusher_desired.x[i] != 0
            or pusher_desired.y[i] != 0
            or pusher_desired.theta[i] != 0
        ):
            first_non_zero_idx = i
            break

    initial_state = np.array(
        [
            pusher_desired.x[first_non_zero_idx],
            pusher_desired.y[first_non_zero_idx],
            pusher_desired.theta[first_non_zero_idx],
        ]
    )
    assert not np.allclose(initial_state, np.array([0.0, 0.0, 0.0]))

    for i in range(first_non_zero_idx + 1, length):
        state = np.array(
            [pusher_desired.x[i], pusher_desired.y[i], pusher_desired.theta[i]]
        )
        if not np.allclose(state, initial_state):
            return i

    return None


def _is_ik_fail(log_path, max_failures=5):
    with open(log_path, "r") as f:
        line = f.readline()
        if len(line) != 0:
            ik_fails = int(line.rsplit(" ", 1)[-1])
            if ik_fails > max_failures:
                return True
    return False


def _get_closest_index(arr, t, start_idx=None, end_idx=None):
    """Returns index of arr that is closest to t."""

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(arr)

    min_diff = float("inf")
    min_idx = -1
    eps = 1e-4
    for i in range(start_idx, end_idx):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < eps:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i


def _compute_angular_speed(time, orientation):
    dt = np.diff(time)
    dtheta = np.diff(orientation)
    angular_speed = abs(dtheta / dt)

    # Remove sharp angular velocity at beginning
    first_zero_idx = -1
    for i in range(len(angular_speed)):
        if np.allclose(angular_speed[i], 0.0):
            first_zero_idx = i
            break

    return angular_speed[first_zero_idx:]


# Function to identify high angular speed moments
def _has_high_angular_speed(slider_desired, threshold, window_size):
    if threshold is None:
        return False

    angular_speed = _compute_angular_speed(slider_desired.t, slider_desired.theta)
    angular_speed_cumsum = np.cumsum(angular_speed)
    max_window_avg = -1
    ret = False
    for i in range(len(angular_speed_cumsum) - window_size):
        window_avg = (
            angular_speed_cumsum[i + window_size] - angular_speed_cumsum[i]
        ) / window_size
        max_window_avg = max(max_window_avg, window_avg)
        if window_avg > threshold:
            return True
    return False


def _print_data_collection_config_info(data_collection_config: DataCollectionConfig):
    """Output diagnostic info about the data collection configuration."""

    print("This data collection script is configured to perform the following steps.\n")
    step_num = 1
    if data_collection_config.generate_plans:
        print(
            f"{step_num}. Generate new plans in '{data_collection_config.plans_dir}' "
            f"according to the following config:"
        )
        print(data_collection_config.plan_config, end="\n\n")
        step_num += 1
    if data_collection_config.render_plans:
        print(
            f"{step_num}. Render the plans in '{data_collection_config.plans_dir}' "
            f"to '{data_collection_config.rendered_plans_dir}'\n"
        )
        step_num += 1
    if data_collection_config.convert_to_zarr:
        print(
            f"{step_num}. Convert the rendered plans in '{data_collection_config.rendered_plans_dir}' "
            f"to zarr format in '{data_collection_config.zarr_path}'"
        )
        if data_collection_config.convert_to_zarr_reduce:
            print(
                "Converting to zarr in 'reduce' mode (i.e. performing the reduce step of map-reduce)"
            )
            print(
                "The 'convert_to_zarr_reduce = True' flag is usually only set for Supercloud runs."
            )
        print()
        step_num += 1


def _print_sim_config_info(sim_config: PlanarPushingSimConfig):
    """Output diagnostic info about the simulation configuration."""

    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")
    print()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[3].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    ## Parse Configs
    sim_config: PlanarPushingSimConfig = PlanarPushingSimConfig.from_yaml(cfg)
    _print_sim_config_info(sim_config)

    data_collection_config: DataCollectionConfig = hydra.utils.instantiate(
        cfg.data_collection_config
    )
    _print_data_collection_config_info(data_collection_config)

    convert_to_zarr(
        rendered_plans_dirs=[
            "trajectories_rendered/test_convex/test_cross",
            "trajectories_rendered/test_convex/test_cross",
        ],
        zarr_path="trajectories_rendered/test_convex/test_cross.zarr",
        sim_config=sim_config,
        data_collection_config=data_collection_config,
        debug=False,
    )


if __name__ == "__main__":
    main()
