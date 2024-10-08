import logging
import os
from datetime import datetime
from typing import List, Literal, Optional, Tuple

import numpy as np

from planning_through_contact.experiments.ablation_study.planar_pushing_ablation import (
    run_ablation,
)
from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    ContactConfig,
    ContactCost,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.utils import (
    get_plan_start_and_goals_to_point,
)
from planning_through_contact.tools.utils import PhysicalProperties


def create_output_folder(
    output_dir: str, slider_type: str, traj_number: Optional[int]
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"{output_dir}/run_{get_time_as_str()}_{slider_type}"
    if traj_number is not None:
        folder_name += f"_traj_{traj_number}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name


def get_time_as_str() -> str:
    current_time = datetime.now()
    # For example, YYYYMMDDHHMMSS format
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time


def get_box(mass) -> RigidBody:
    box_geometry = Box2d(width=0.1, height=0.1)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def get_tee(mass) -> RigidBody:
    body = RigidBody("t_pusher", TPusher2d(), mass)
    return body


def get_arbitrary(
    arbitrary_shape_pickle_path: str, mass: float, com: np.ndarray = None
) -> RigidBody:
    "com assumes uniform density if None."
    body = RigidBody(
        "arbitrary", ArbitraryShape2D(arbitrary_shape_pickle_path, com), mass
    )
    return body


def get_sugar_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.106, height=0.185)
    slider = RigidBody("sugar_box", box_geometry, mass)
    return slider


def get_default_contact_cost() -> ContactCost:
    contact_cost = ContactCost(
        keypoint_arc_length=10.0,
        force_regularization=100000.0,  # NOTE: This is multiplied by 1e-4 because we have forces in other units in the optimization problem
        keypoint_velocity_regularization=100.0,
        ang_velocity_regularization=None,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.1,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=10.0,
        time=None,
    )
    return non_collision_cost


def get_hardware_contact_cost() -> ContactCost:
    """
    A custom cost for hardware,
    which empically generates plans that respect robot velocity
    limits etc.
    """
    contact_cost = ContactCost(
        keypoint_arc_length=10.0,
        force_regularization=100000.0,
        keypoint_velocity_regularization=100.0,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_hardware_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.25,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=5.0,
        time=None,
    )
    return non_collision_cost


def get_default_plan_config(
    slider_type: Literal["box", "sugar_box", "tee", "arbitrary"] = "box",
    arbitrary_shape_pickle_path: str = "",
    slider_physical_properties: PhysicalProperties = None,
    pusher_radius: float = 0.015,
    time_contact: float = 2.0,
    time_non_collision: float = 4.0,
    workspace: Optional[PlanarPushingWorkspace] = None,
    hardware: bool = False,
) -> PlanarPlanConfig:
    mass = (
        0.1 if slider_physical_properties is None else slider_physical_properties.mass
    )
    com = (
        None
        if slider_physical_properties is None
        else slider_physical_properties.center_of_mass
    )
    if slider_physical_properties is None:
        logging.warning("Using default mass of 0.1 kg for the slider.")
    if slider_type == "box":
        slider = get_box(mass)
    elif slider_type == "sugar_box":
        slider = get_sugar_box(mass)
    elif slider_type == "tee":
        slider = get_tee(mass)
    elif slider_type == "arbitrary":
        slider = get_arbitrary(arbitrary_shape_pickle_path, mass, com)
    else:
        raise NotImplementedError(f"Slider type {slider_type} not supported")

    if hardware:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.05,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )

        contact_cost = get_hardware_contact_cost()
        non_collision_cost = get_hardware_non_collision_cost()
        lam_buffer = 0.25
        contact_config = ContactConfig(
            cost=contact_cost, lam_min=lam_buffer, lam_max=1 - lam_buffer
        )
        time_contact = 5.0
        time_non_collision = 2.0

        num_knot_points_non_collision = 5
        num_knot_points_contact = 3
    else:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.1,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )
        contact_cost = get_default_contact_cost()
        non_collision_cost = get_default_non_collision_cost()
        lam_buffer = 0.0
        contact_config = ContactConfig(
            cost=contact_cost, lam_min=lam_buffer, lam_max=1 - lam_buffer
        )

        time_contact = 4.0
        time_non_collision = 2.0

        num_knot_points_non_collision = 3
        num_knot_points_contact = 3

    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        num_knot_points_contact=num_knot_points_contact,
        num_knot_points_non_collision=num_knot_points_non_collision,
        use_band_sparsity=True,
        contact_config=contact_config,
        non_collision_cost=non_collision_cost,
        continuity_on_pusher_velocity=True,
        allow_teleportation=False,
        time_in_contact=time_contact,
        time_non_collision=time_non_collision,
        workspace=workspace,
    )

    return plan_cfg


def get_default_solver_params(
    debug: bool = False, clarabel: bool = False
) -> PlanarSolverParams:
    solver_params = PlanarSolverParams(
        measure_solve_time=debug,
        rounding_steps=100,
        print_flows=False,
        solver="mosek" if not clarabel else "clarabel",
        print_solver_output=debug,
        save_solver_output=False,
        print_rounding_details=debug,
        print_path=False,
        print_cost=debug,
        assert_result=False,
        assert_nan_values=True,
        nonl_round_major_feas_tol=1e-5,
        nonl_round_minor_feas_tol=1e-5,
        nonl_round_opt_tol=1e-5,
        max_mosek_solve_time=3000.0,
    )
    return solver_params


def get_hardware_plans(
    hardware_seed: int, config: PlanarPlanConfig
) -> List[PlanarPushingStartAndGoal]:
    """
    Generates a collection of plans that can be run on our hardware setup with the Kuka Iiwa, with the right workspace
    and origin.
    """
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=0.35,
            height=0.5,
            center=np.array([0.575, 0.0]),
            buffer=0,
        ),
    )

    num_trajs = 30
    plans = get_plan_start_and_goals_to_point(
        hardware_seed,
        num_trajs,
        workspace,
        config,
        (0.575, -0.04285714),
        limit_rotations=False,
    )

    return plans


def get_default_experiment_plans(
    seed: int, num_trajs: int, config: PlanarPlanConfig, workspace_size: float = 0.6
) -> List[PlanarPushingStartAndGoal]:
    """
    Generates a collection of random initial configurations with the origin as the target
    configuration.
    """
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=workspace_size,
            height=workspace_size,
            center=np.array([0.0, 0.0]),
            buffer=0,
        ),
    )

    plans = get_plan_start_and_goals_to_point(
        seed,
        num_trajs,
        workspace,
        config,
        (0.0, 0.0),
        limit_rotations=False,
    )

    return plans


def run_ablation_with_default_config(
    slider_type: Literal["box", "sugar_box", "tee"],
    pusher_radius: float,
    integration_constant: float,
    num_experiments: int,
    arc_length_weight: Optional[float] = None,
    filename: Optional[str] = None,
) -> None:
    config = get_default_plan_config(
        slider_type, pusher_radius, integration_constant, arc_length_weight  # type: ignore
    )
    solver_params = get_default_solver_params()
    run_ablation(config, solver_params, num_experiments, filename)  # type: ignore


def get_baseline_comparison_costs() -> Tuple[ContactCost, NonCollisionCost]:
    contact_cost = get_default_contact_cost()
    non_collision_cost = get_default_non_collision_cost()

    return contact_cost, non_collision_cost


def get_default_baseline_solver_params() -> PlanarSolverParams:
    solver_params = get_default_solver_params()

    solver_params.nonl_rounding_save_solver_output = False
    solver_params.print_cost = False
    return solver_params


def get_baseline_comparison_configs(
    slider_type: Literal["box", "sugar_box", "tee"] = "sugar_box",
    only_minimize_arc_lengths: bool = False,
    use_velocity_limits: bool = True,
) -> Tuple[PlanarPlanConfig, PlanarSolverParams]:
    config = get_default_plan_config(slider_type=slider_type)
    # Make the dt's for contact and noncontact the same
    dt = 0.25
    config.num_knot_points_contact = 3
    config.time_in_contact = config.num_knot_points_contact * dt
    config.num_knot_points_non_collision = 4
    config.time_non_collision = config.num_knot_points_non_collision * dt

    if only_minimize_arc_lengths:
        # These parameters seem to give the best posed optimization problem for
        # the direct method, without impacting the cost we are measuring it on
        # (which is the total arc length of pusher + slider)
        config.contact_config.cost.force_regularization = 10
        config.contact_config.cost.keypoint_velocity_regularization = 0.1
        config.non_collision_cost.pusher_velocity_regularization = 0.1

        config.contact_config.cost.time = 1
        config.non_collision_cost.distance_to_object_socp = 0.1
        config.non_collision_cost.pusher_arc_length = 1
        config.contact_config.cost.keypoint_arc_length = 1
    else:
        config.contact_config.cost.force_regularization = 1000
        config.contact_config.cost.keypoint_velocity_regularization = 10
        config.non_collision_cost.pusher_arc_length = 1
        config.non_collision_cost.pusher_velocity_regularization = 5

    if use_velocity_limits:
        # The direct method needs velocity limits, otherwise it doesn't work
        VEL_LIMIT = 0.3  # m/s
        ANG_VEL_LIMIT = (2 * np.pi) / 4
        config.contact_config.slider_velocity_constraint = VEL_LIMIT
        config.non_collision_cost.pusher_velocity_constraint = VEL_LIMIT
        config.contact_config.slider_rot_velocity_constraint = ANG_VEL_LIMIT

    # No force scaling
    # TODO: The force scaling will be removed entirely
    config.dynamics_config.force_scale = 1

    config.contact_config.lam_min = 0
    config.contact_config.lam_max = 1

    assert config.dt_contact == config.dt_non_collision

    solver_params = get_default_baseline_solver_params()

    return config, solver_params
