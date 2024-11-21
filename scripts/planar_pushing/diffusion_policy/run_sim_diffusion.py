import importlib
import logging
import os
import pathlib
import signal
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicySource,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.output_feedback_table_environment import (
    OutputFeedbackTableEnvironment,
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
    config_path=str(pathlib.Path(__file__).parents[3].joinpath("config", "sim_config")),
)
def run_sim(cfg: OmegaConf):
    logging.basicConfig(level=logging.INFO)
    if "save_logs" in cfg.diffusion_policy_config:
        save_logs = cfg.diffusion_policy_config.save_logs
    else:
        save_logs = False

    # start meshcat
    print(f"station meshcat")
    station_meshcat = StartMeshcat()

    # load sim_config
    sim_config = PlanarPushingSimConfig.from_yaml(cfg)
    print(f"Initial finger pose: {sim_config.pusher_start_pose}")
    print(f"Target slider pose: {sim_config.slider_goal_pose}")

    if cfg.slider_type == "arbitrary":
        # create arbitrary shape sdf file
        create_arbitrary_shape_sdf_file(cfg, sim_config)

    # Diffusion Policy source
    position_source = DiffusionPolicySource(sim_config.diffusion_policy_config)
    if save_logs:
        pickled_logs_dir = "pickled_logs"  # TODO: make this a config option

        def signal_handler(sig, frame):
            print("Received signal: ", sig)

            if not os.path.exists(pickled_logs_dir):
                os.makedirs(pickled_logs_dir)
            num_files = len(
                [
                    file
                    for file in os.listdir(pickled_logs_dir)
                    if os.path.isfile(os.path.join(pickled_logs_dir, file))
                ]
            )
            position_source._diffusion_policy_controller.save_logs_to_file(
                f"{pickled_logs_dir}/{num_files}.pkl"
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    # Set up position controller
    # TODO: load with hydra instead (currently giving camera config errors)
    module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
    robot_system_class = getattr(importlib.import_module(module_name), class_name)
    position_controller: RobotSystemBase = robot_system_class(
        sim_config=sim_config, meshcat=station_meshcat
    )

    # Set up environment
    environment = OutputFeedbackTableEnvironment(
        desired_position_source=position_source,
        robot_system=position_controller,
        sim_config=sim_config,
        station_meshcat=station_meshcat,
        arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
    )

    environment.export_diagram("diffusion_environment_diagram.pdf")

    # Configure sim and recording
    recording_name = "diffusion_policy_roll_out.html"
    # environment.export_diagram("diffusion_environment_diagram.pdf")
    if sim_config.multi_run_config is None:
        end_time = 100.0
        seed = "N/A (no multi_run_config seed provided)"
    else:
        num_runs = sim_config.multi_run_config.num_runs
        max_attempt_duration = sim_config.multi_run_config.max_attempt_duration
        seed = sim_config.multi_run_config.seed
        end_time = num_runs * max_attempt_duration

    successful_idx, save_dir = environment.simulate(
        end_time, recording_file=recording_name
    )

    if save_logs:
        if not os.path.exists(pickled_logs_dir):
            os.makedirs(pickled_logs_dir)
        num_files = len(
            [
                file
                for file in os.listdir(pickled_logs_dir)
                if os.path.isfile(os.path.join(pickled_logs_dir, file))
            ]
        )
        position_source._diffusion_policy_controller.save_logs_to_file(
            f"pickled_logs/{num_files}.pkl"
        )

    # Update logs and save config file
    OmegaConf.save(cfg, f"{save_dir}/sim_config.yaml")
    with open(f"{cfg.log_dir}/checkpoint_statistics.txt", "a") as f:
        f.write(f"{sim_config.diffusion_policy_config.checkpoint}\n")
        f.write(f"Seed: {seed}\n")
        f.write(
            f"Success ratio: {len(successful_idx)} / {num_runs} = {100.0*len(successful_idx) / num_runs:.3f}%\n"
        )
        f.write(f"Success_idx: {successful_idx}\n")
        f.write(f"trans_tol: {cfg.multi_run_config.trans_tol}\n")
        f.write(f"rot_tol: {cfg.multi_run_config.rot_tol}\n")
        f.write(
            f"evaluate_final_pusher_position: {cfg.multi_run_config.evaluate_final_pusher_position}\n"
        )
        f.write(
            f"evaluate_final_slider_rotation: {cfg.multi_run_config.evaluate_final_slider_rotation}\n"
        )
        f.write(f"Save dir: {save_dir}\n")
        f.write("\n")

    if cfg.slider_type == "arbitrary":
        # Remove the sdf file.
        sdf_path = get_slider_sdf_path(sim_config, models_folder)
        if os.path.exists(sdf_path):
            os.remove(sdf_path)


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/diffusion_policy/planar_pushing/run_sim_diffusion.py --config-name=actuated_cylinder_sim_config.yaml
    """
    run_sim()
