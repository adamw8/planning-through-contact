import importlib
import logging
import os
import pathlib
import pickle
import random
import shutil
import time
from enum import Enum
from typing import Optional

import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from pydrake.all import HPolyhedron, StartMeshcat, VPolytope

from planning_through_contact.experiments.utils import get_default_plan_config
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    PlanarPushingWorkspace,
)
from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicySource,
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
from planning_through_contact.visualize.analysis import (
    CombinedPlanarPushingLogs,
    PlanarPushingLog,
)


class SimSimEval:
    def __init__(self, cfg: OmegaConf):
        # start meshcat
        print(f"Station meshcat")
        station_meshcat = StartMeshcat()

        # load sim_config
        self.cfg = cfg
        self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        self.multi_run_config = self.sim_config.multi_run_config
        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        print(f"Initial pusher pose: {self.pusher_start_pose}")
        print(f"Target slider pose: {self.slider_goal_pose}")

        # Set up random seeds
        random.seed(self.multi_run_config.seed)
        np.random.seed(self.multi_run_config.seed)

        self.workspace = self.multi_run_config.workspace
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

        # Diffusion Policy
        position_source = DiffusionPolicySource(self.sim_config.diffusion_policy_config)

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
        self.environment.export_diagram("sim_sim_environment.pdf")

        # Random initial conditoin
        self.reset_environment()

        # Useful variables for querying mbp
        self.plant = self.environment._plant
        self.mbp_context = self.environment.mbp_context
        self.pusher_body = self.plant.GetBodyByName("pusher")
        self.slider_model_instance = self.environment._slider_model_instance

        # Success_criteria
        valid_success_criteria = ["tolerance", "convex_hull"]
        self.success_criteria = self.multi_run_config.success_criteria
        assert self.success_criteria in valid_success_criteria

        if self.success_criteria == "convex_hull":
            dataset_path = self.multi_run_config.dataset_path
            self.pusher_goal_convex_hull = self.get_pusher_goal_polyhedron(dataset_path)
            self.slider_goal_convex_hull = self.get_slider_goal_polyhedron(dataset_path)

    def simulate_environment(
        self,
        end_time: float,
        recording_file: Optional[str] = None,
    ):
        # Loop variables
        time_step = self.sim_config.time_step * 10
        t = time_step
        last_reset_time = t
        num_successful_trials = 0
        num_completed_trials = 0

        # Simulate
        self.environment.visualize_desired_slider_pose()
        self.environment.visualize_desired_pusher_pose()
        while t < end_time:
            self.environment._simulator.AdvanceTo(t)

            reset_environment = False
            success = False

            # Check for failure
            if self.check_success():
                success = True
                reset_environment = True
            # Check for failure
            elif self.check_failure(t, last_reset_time):
                reset_environment = True

            # Reset environment
            if reset_environment:
                self.reset_environment()
                last_reset_time = t
                num_completed_trials += 1

                if success:
                    num_successful_trials += 1

            # Finished Eval
            if num_completed_trials >= self.multi_run_config.num_runs:
                break

            # Loop updates
            t += time_step
            t = round(t / time_step) * time_step

        # save the logs somewhere

    def check_success(self):
        if self.success_criteria == "tolerance":
            return self._check_success_tolerance()
        elif self.success_criteria == "convex_hull":
            return self._check_success_convex_hull()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _check_success_tolerance(self):
        # slider
        slider_pose = self.get_slider_pose()
        slider_goal_pose = self.sim_config.slider_goal_pose
        slider_error = slider_goal_pose.vector() - slider_pose.vector()
        reached_goal_slider_position = (
            np.linalg.norm(slider_error[:2]) <= self.multi_run_config.trans_tol
        )
        reached_goal_slider_orientation = (
            np.abs(slider_error[2]) <= self.multi_run_config.rot_tol
        )

        # pusher
        pusher_pose = self.get_pusher_pose()
        pusher_goal_pose = self.sim_config.pusher_start_pose
        pusher_error = pusher_goal_pose.vector() - pusher_pose.vector()
        reached_goal_pusher_position = (
            np.linalg.norm(pusher_error[:2]) <= self.multi_run_config.trans_tol
        )

        if not reached_goal_slider_position:
            return False
        if (
            self.multi_run_config.evaluate_final_slider_rotation
            and not reached_goal_slider_orientation
        ):
            return False
        if (
            self.multi_run_config.evaluate_final_pusher_position
            and not reached_goal_pusher_position
        ):
            return False
        return True

    def _check_success_convex_hull(self):
        slider_pose = self.get_slider_pose().vector()
        pusher_position = self.get_pusher_pose().vector()[:2]

        slider_success = self.is_contained(slider_pose, self.slider_goal_convex_hull)
        pusher_success = self.is_contained(
            pusher_position, self.pusher_goal_convex_hull
        )
        print(f"Slider success: {slider_success}, Pusher success: {pusher_success}")
        return slider_success and pusher_success

    def check_failure(self, t, last_reset_time):
        # Check timeout
        duration = t - last_reset_time
        if duration > self.multi_run_config.max_attempt_duration:
            return True

        # Check if slider is on table
        z_value = self.plant.GetPositions(self.mbp_context, self.slider_model_instance)[
            -1
        ]
        if z_value < 0.0:
            return True

        # No immediate failures
        return False

    # TODO: write reset function for diffusion policy
    def reset_environment(self):
        seed = int(1e6 * time.time() % 1e6)
        np.random.seed(seed)
        slider_geometry = self.sim_config.dynamics_config.slider.geometry
        slider_pose = get_slider_pose_within_workspace(
            self.workspace, slider_geometry, self.pusher_start_pose, self.plan_config
        )

        self.environment.reset(
            np.array([0.6202, 1.0135, -0.5873, -1.4182, 0.6449, 0.8986, 2.9879]),
            slider_pose,
            self.pusher_start_pose,
        )

    def get_planar_pushing_log(self, vector_log, traj_start_time):
        start_idx = 0
        sample_times = vector_log.sample_times()
        while sample_times[start_idx] < traj_start_time:
            start_idx += 1

        t = sample_times[start_idx:] - sample_times[start_idx]
        nan_array = np.array([float("nan") for _ in t])
        return PlanarPushingLog(
            t=t,
            x=vector_log.data()[0, start_idx:],
            y=vector_log.data()[1, start_idx:],
            theta=vector_log.data()[2, start_idx:],
            lam=nan_array,
            c_n=nan_array,
            c_f=nan_array,
            lam_dot=nan_array,
        )

    def get_pusher_pose(self):
        pusher_position = self.plant.EvalBodyPoseInWorld(
            self.mbp_context, self.pusher_body
        ).translation()
        return PlanarPose(pusher_position[0], pusher_position[1], 0.0)

    def get_slider_pose(self):
        slider_pose = self.plant.GetPositions(
            self.mbp_context, self.slider_model_instance
        )
        return PlanarPose.from_generalized_coords(slider_pose)

    def get_pusher_goal_polyhedron(self, dataset_path):
        root = zarr.open(dataset_path, mode="r")
        indices = np.array(root["meta/episode_ends"]) - 1
        state = np.array(root["data/state"])
        final_positions = state[indices][:, :2]
        return HPolyhedron(VPolytope(final_positions.transpose()))

    def get_slider_goal_polyhedron(self, dataset_path):
        root = zarr.open(dataset_path, mode="r")
        indices = np.array(root["meta/episode_ends"]) - 1
        state = np.array(root["data/slider_state"])
        final_states = state[indices]
        return HPolyhedron(VPolytope(final_states.transpose()))

    def is_contained(self, point, polyhedron):
        A, b = polyhedron.A(), polyhedron.b()
        return np.all(A @ point <= b)


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


# TODO: rewrite the main function
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[2].joinpath("config", "sim_config")),
)
def main(cfg: OmegaConf):
    sim_sim_eval = SimSimEval(cfg)
    sim_sim_eval.simulate_environment(float("inf"))


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/diffusion_policy/planar_pushing/run_gamepad_teleop.py --config-dir <dir> --config-name <file>
    """
    main()
