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


class GamepadDataCollection:
    def __init__(self, cfg: OmegaConf):
        # start meshcat
        print(f"Station meshcat")
        station_meshcat = StartMeshcat()

        # load sim_config
        self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        print(f"Initial pusher pose: {self.pusher_start_pose}")
        print(f"Target slider pose: {self.slider_goal_pose}")

        self.workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=0.5,
                height=0.35,
                center=np.array([self.slider_goal_pose.x, self.slider_goal_pose.y]),
                buffer=0,
            ),
        )
        self.plan_config = get_default_plan_config(
            slider_type=self.sim_config.slider.name
            if self.sim_config.slider.name != "t_pusher"
            else "tee",
            arbitrary_shape_pickle_path=self.sim_config.arbitrary_shape_pickle_path,
            pusher_radius=0.015,
            hardware=False,
        )

        if cfg.slider_type == "arbitrary":
            # create arbitrary shape sdf file
            create_arbitrary_shape_sdf_file(cfg, self.sim_config)

        # Gamepad Controller Source
        position_source = GamepadControllerSource(station_meshcat)

        # Set up position controller
        module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
        robot_system_class = getattr(importlib.import_module(module_name), class_name)
        position_controller: RobotSystemBase = robot_system_class(
            sim_config=self.sim_config, meshcat=station_meshcat
        )

        # Set up environment
        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=station_meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        self.environment.export_diagram("gamepad_teleop_environment.pdf")

        # fsm state
        self.fsm_state = FSMState.REGULAR
        self.traj_start_time = 0.0

    def simulate_environment(
        self,
        end_time: float,
        recording_file: Optional[str] = None,
    ):
        prev_button_values = self.environment.get_button_values()
        time_step = self.sim_config.time_step * 10
        self.environment.visualize_desired_slider_pose()
        t = time_step

        while t < end_time:
            self.environment._simulator.AdvanceTo(t)

            # Get pressed buttons
            button_values = self.environment.get_button_values()
            pressed_buttons = self.get_pressed_buttons(
                prev_button_values, button_values
            )

            # FSM logic
            self.fsm_state, self.traj_start_time = self.fsm_logic(
                self.fsm_state, pressed_buttons, t, self.traj_start_time
            )

            # Loop updates
            t += time_step
            t = round(t / time_step) * time_step
            prev_button_values = button_values

    def fsm_logic(self, fsm_state, pressed_buttons, curr_time, traj_start_time):
        pressed_A = pressed_buttons["A"]
        pressed_B = pressed_buttons["B"]  # Reset environment
        if fsm_state == FSMState.REGULAR:
            if pressed_A:
                print_blue(f"Entering data collection mode at time: {curr_time:.2f}")
                return FSMState.DATA_COLLECTION, curr_time
            if pressed_B:
                self.reset_environment()
                print_blue("Reset environment. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
        elif fsm_state == FSMState.DATA_COLLECTION:
            if pressed_A:
                self.save_trajectory()
                print_blue("Saved trajectory. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
            elif pressed_B:
                self.delete_trajectory()
                print_blue("Deleted trajectory. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
        return fsm_state, traj_start_time

    def reset_environment(self):
        seed = int(1e6 * time.time() % 1e6)
        np.random.seed(seed)
        slider_geometry = self.sim_config.dynamics_config.slider.geometry
        slider_pose = get_slider_pose_within_workspace(
            self.workspace, slider_geometry, self.pusher_start_pose, self.plan_config
        )

        self.environment.reset(
            np.array([0.6146, 1.0226, -0.5869, -1.4031, 0.6442, 0.9059, 2.9904]),
            slider_pose,
            self.pusher_start_pose,
        )

    def save_trajectory(self):
        self.reset_environment()

    def delete_trajectory(self):
        self.reset_environment()

    def get_pressed_buttons(self, prev_button_values, button_values):
        pressed_buttons = {}
        for button, value in button_values.items():
            if value and not prev_button_values[button]:
                pressed_buttons[button] = True
            else:
                pressed_buttons[button] = False
        return pressed_buttons


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    gamepad_data_collection = GamepadDataCollection(cfg)
    gamepad_data_collection.simulate_environment(float("inf"))


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/diffusion_policy/planar_pushing/run_gamepad_teleop.py --config-dir <dir> --config-name <file>
    """
    main()
