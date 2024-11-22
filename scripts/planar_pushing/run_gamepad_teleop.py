import importlib
import logging
import os
import pathlib
import random
import time
from enum import Enum
from typing import Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.gamepad_controller_source import (
    GamepadControllerSource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.simulated_real_table_environment import (
    SimulatedRealTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    create_arbitrary_shape_sdf_file,
    get_slider_pose_within_workspace,
    get_slider_sdf_path,
    models_folder,
)


class FSMState(Enum):
    REGULAR = "regular"
    DATA_COLLECTION = "data collection"


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def run_sim(cfg: OmegaConf):
    # logging.getLogger('drake').setLevel(logging.INFO)
    # logging.getLogger('root').setLevel(logging.INFO)
    # logging.getLogger(
    #     'planning_through_contact.simulation.systems.joint_velocity_clamp'
    # ).setLevel(logging.INFO)

    # Set seed
    seed = int(1e6 * time.time() % 1e6)
    np.random.seed(seed)
    random.seed(seed)

    # start meshcat
    print(f"Station meshcat")
    station_meshcat = StartMeshcat()

    # load sim_config
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)
    print(f"Initial pusher pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")

    if cfg.slider_type == "arbitrary":
        # create arbitrary shape sdf file
        create_arbitrary_shape_sdf_file(cfg, sim_config)

    # Gamepad Controller Source
    position_source = GamepadControllerSource(station_meshcat)

    # Set up position controller
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, meshcat=station_meshcat
    )

    # Set up environment
    environment = SimulatedRealTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
        arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
    )
    environment.export_diagram("gamepad_teleop_environment.pdf")
    simulate_environment(environment, float("inf"), sim_config)


# TODO: handle recording file and data saving
def simulate_environment(
    environment: SimulatedRealTableEnvironment,
    end_time: float,
    sim_config,
    recording_file: Optional[str] = None,
):
    fsm_state = FSMState.REGULAR
    traj_start_time = 0.0

    prev_button_values = environment.get_button_values()
    time_step = environment._sim_config.time_step * 100
    environment.visualize_desired_slider_pose()
    t = time_step

    while t < end_time:
        environment._simulator.AdvanceTo(t)

        # Get pressed buttons
        button_values = environment.get_button_values()
        pressed_buttons = get_pressed_buttons(prev_button_values, button_values)

        # FSM logic
        fsm_state, traj_start_time = fsm_logic(
            fsm_state, pressed_buttons, t, traj_start_time, environment, sim_config
        )

        # Print every 5 seconds
        # if t % 5 == 0:
        #     print(f"Simulation Time: {t}s")

        # Loop updates
        t += time_step
        t = round(t / time_step) * time_step
        prev_button_values = button_values


def fsm_logic(
    fsm_state, pressed_buttons, curr_time, traj_start_time, environment, sim_config
):
    pressed_A = pressed_buttons["A"]
    pressed_B = pressed_buttons["B"]  # Reset environment
    if fsm_state == FSMState.REGULAR:
        if pressed_A:
            print("Entering data collection mode at time: ", curr_time)
            return FSMState.DATA_COLLECTION, curr_time
        if pressed_B:
            reset_environment(environment, sim_config)
            print("Reset environment. Entering regular mode.")
            return FSMState.REGULAR, traj_start_time
    elif fsm_state == FSMState.DATA_COLLECTION:
        if pressed_A:
            save_trajectory(environment, traj_start_time, sim_config)
            print("Saved trajectory. Entering regular mode.")
            return FSMState.REGULAR, traj_start_time
        elif pressed_B:
            delete_trajectory(environment, sim_config)
            print("Deleted trajectory. Entering regular mode.")
            return FSMState.REGULAR, traj_start_time
    return fsm_state, traj_start_time


def reset_environment(environment, sim_config):
    # Set up plan config
    pusher_pose = sim_config.pusher_start_pose
    slider_goal_pose = sim_config.slider_goal_pose
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=0.5,
            height=0.35,
            center=np.array([slider_goal_pose.x, slider_goal_pose.y]),
            buffer=0,
        ),
    )
    plan_config = get_default_plan_config(
        slider_type=sim_config.slider.name
        if sim_config.slider.name != "t_pusher"
        else "tee",
        arbitrary_shape_pickle_path=sim_config.arbitrary_shape_pickle_path,
        pusher_radius=0.015,
        hardware=False,
    )

    seed = int(1e6 * time.time() % 1e6)
    np.random.seed(seed)
    slider_geometry = sim_config.dynamics_config.slider.geometry
    slider_pose = get_slider_pose_within_workspace(
        workspace, slider_geometry, pusher_pose, plan_config
    )

    environment.reset(
        np.array([0.6146, 1.0226, -0.5869, -1.4031, 0.6442, 0.9059, 2.9904]),
        slider_pose,
        pusher_pose,
    )


def save_trajectory(environment, start_time, sim_config):
    reset_environment(environment, sim_config)


def delete_trajectory(environment, sim_config):
    print("Deleting trajectory. Entering regular mode.")
    reset_environment(environment, sim_config)


def get_pressed_buttons(prev_button_values, button_values):
    pressed_buttons = {}
    for button, value in button_values.items():
        if value and not prev_button_values[button]:
            pressed_buttons[button] = True
        else:
            pressed_buttons[button] = False
    return pressed_buttons


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/diffusion_policy/planar_pushing/run_gamepad_teleop.py --config-dir <dir> --config-name <file>
    """
    run_sim()
