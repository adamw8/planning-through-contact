import importlib
import logging
import os
import pathlib
from typing import Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

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
    get_slider_sdf_path,
    models_folder,
)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def run_sim(cfg: OmegaConf):
    logging.basicConfig(level=logging.INFO)

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
    simulate_environment(environment, end_time=float("inf"))


# TODO: handle recording file and data saving
def simulate_environment(
    environment: SimulatedRealTableEnvironment,
    end_time: float,
    recording_file: Optional[str] = None,
):
    prev_button_values = environment.get_button_values()
    time_step = environment._sim_config.time_step * 100
    environment.visualize_desired_slider_pose()
    t = time_step
    while t < end_time:
        environment._simulator.AdvanceTo(t)

        # Get pressed buttons
        button_values = environment.get_button_values()
        pressed_buttons = get_pressed_buttons(prev_button_values, button_values)

        # Print every 5 seconds
        if t % 5 == 0:
            print(f"Simulation Time: {t}s")

        # Loop updates
        t += time_step
        t = round(t / time_step) * time_step
        prev_button_values = button_values


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
