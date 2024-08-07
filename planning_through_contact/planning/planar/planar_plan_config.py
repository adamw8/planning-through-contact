from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody


@dataclass
class BoxWorkspace:
    width: float = 0.5
    height: float = 0.5
    center: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    buffer: float = 0.0

    @property
    def x_min(self) -> float:
        return self.center[0] - self.width / 2 - self.buffer

    @property
    def x_max(self) -> float:
        return self.center[0] + self.width / 2 + self.buffer

    @property
    def y_min(self) -> float:
        return self.center[1] - self.height / 2 - self.buffer

    @property
    def y_max(self) -> float:
        return self.center[1] + self.height / 2 + self.buffer

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lb = np.array([self.x_min, self.y_min], dtype=np.float64)
        ub = np.array([self.x_max, self.y_max], dtype=np.float64)
        return lb, ub

    def new_workspace_with_buffer(self, new_buffer: float) -> "BoxWorkspace":
        return BoxWorkspace(self.width, self.height, self.center, new_buffer)


@dataclass
class PlanarPushingWorkspace:
    slider: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(
            width=1.0, height=1.0, center=np.array([0.0, 0.0]), buffer=0.0
        )
    )


@dataclass
class SliderPusherSystemConfig:
    slider: RigidBody = field(
        default_factory=lambda: RigidBody(
            name="box", geometry=Box2d(width=0.15, height=0.15), mass=0.1
        )
    )
    pusher_radius: float = 0.015
    friction_coeff_table_slider: float = 0.5
    friction_coeff_slider_pusher: float = 0.1
    grav_acc: float = 9.81
    integration_constant: float = 0.6
    force_scale: float = (
        0.01  # Scaling of the forces to make the optimization program better posed
    )

    @cached_property
    def f_max(self) -> float:
        return self.friction_coeff_table_slider * self.grav_acc * self.slider.mass

    @cached_property
    def max_contact_radius(self) -> float:
        geometry = self.slider.geometry
        if (
            isinstance(geometry, Box2d)
            or isinstance(geometry, TPusher2d)
            or isinstance(geometry, ArbitraryShape2D)
        ):
            return np.sqrt((geometry.width / 2) ** 2 + (geometry.height) ** 2)
        else:
            raise NotImplementedError(
                f"max_contact_radius for {type(geometry)} is not implemented"
            )

    @cached_property
    def tau_max(self) -> float:
        return self.f_max * self.max_contact_radius * self.integration_constant

    @cached_property
    def ellipsoidal_limit_surface(self) -> npt.NDArray[np.float64]:
        D = np.diag([1 / self.f_max**2, 1 / self.f_max**2, 1 / self.tau_max**2])
        return D

    @cached_property
    def limit_surface_const(self) -> float:
        return (self.max_contact_radius * self.integration_constant) ** -2

    def __eq__(self, other: "SliderPusherSystemConfig") -> bool:
        return (
            self.slider == other.slider
            and self.pusher_radius == other.pusher_radius
            and self.friction_coeff_table_slider == other.friction_coeff_table_slider
            and self.friction_coeff_slider_pusher == other.friction_coeff_slider_pusher
            and self.grav_acc == other.grav_acc
            and self.integration_constant == other.integration_constant
            and self.force_scale == other.force_scale
        )


@dataclass
class PlanarSolverParams:
    rounding_steps: int = 20
    max_rounding_trials: int = 10000  # number of rounding trials to find paths in the graph BEFORE solving any ConvexRestriction
    gcs_convex_relaxation: bool = True  # NOTE: Currently, there is no way to solve the MISDP, so this must be true
    print_flows: bool = False
    assert_determinants: bool = False  # TODO: Remove this
    assert_result: bool = True
    # Flag to assert that all values on GCS path are not NaN, and that all non-path values are NaN
    # (this can happen if the convex sets are not compact)
    assert_nan_values: bool = True
    print_solver_output: bool = False
    save_solver_output: bool = False
    measure_solve_time: bool = False
    max_mosek_solve_time: float = 300.0
    print_path: bool = False
    print_cost: bool = False
    print_rounding_details: bool = False
    solver: Literal["mosek", "clarabel"] = "mosek"
    get_rounded_and_original_traj: bool = False
    nonl_round_major_feas_tol: float = (
        1e-3  # Feasibility treshold for nonlinear rounding
    )
    nonl_round_minor_feas_tol: float = (
        1e-4  # Feasibility treshold for nonlinear rounding
    )
    nonl_round_opt_tol: float = 1e-4  # Optimality treshold for nonlinear rounding
    nonl_rounding_save_solver_output: bool = False
    # nonl_round_major_feas_tol: float = (
    #     1e-6  # Feasibility treshold for nonlinear rounding
    # )
    # nonl_round_minor_feas_tol: float = (
    #     1e-6  # Feasibility treshold for nonlinear rounding
    # )
    # nonl_round_opt_tol: float = 1e-6  # Optimality treshold for nonlinear rounding
    nonl_round_major_iter_limit: int = 10000  # Max number of major iterations of snopt
    assert_rounding_res: bool = False  # We don't run rounding to optimality
    sol_retrieval: Literal["first_row", "eigenvec"] = "first_row"


@dataclass
class NonCollisionCost:
    distance_to_object_socp: Optional[float] = None
    # NOTE: The single mode is only used to test one non-collision mode at a time
    distance_to_object_socp_single_mode: Optional[float] = None
    pusher_velocity_regularization: Optional[float] = None
    pusher_velocity_constraint: Optional[
        float
    ] = None  # TODO: move this (it is not a cost, as the name of the class entails it should be)
    pusher_arc_length: Optional[float] = None
    time: Optional[float] = None

    @property
    def avoid_object(self) -> bool:
        return (
            self.distance_to_object_socp is not None
            or self.distance_to_object_socp_single_mode is not None
        )

    def __str__(self) -> str:
        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]
        return "\n".join(field_strings)


@dataclass
class ContactCost:
    keypoint_arc_length: Optional[float] = None
    force_regularization: Optional[float] = None
    keypoint_velocity_regularization: Optional[float] = None
    ang_velocity_regularization: Optional[float] = None
    mode_transition_cost: Optional[float] = None
    trace: Optional[float] = None
    time: Optional[float] = None

    def __str__(self) -> str:
        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]
        return "\n".join(field_strings)


# TODO: Refactor this
@dataclass
class ContactConfig:
    cost: ContactCost = field(default_factory=ContactCost)
    # Min and max values for the scaled position of the finger on the face of the slider
    lam_min: Optional[float] = 0.0
    lam_max: Optional[float] = 1.0
    slider_rot_velocity_constraint: Optional[float] = None
    slider_velocity_constraint: Optional[float] = None
    keypoint_velocity_constraint: Optional[float] = None

    def __str__(self) -> str:
        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]
        return "\n".join(field_strings)


@dataclass
class PlanarPushingStartAndGoal:
    slider_initial_pose: PlanarPose
    slider_target_pose: PlanarPose
    pusher_initial_pose: Optional[PlanarPose] = None
    pusher_target_pose: Optional[PlanarPose] = None

    def rotate(self, theta: float) -> "PlanarPushingStartAndGoal":
        new_slider_init = self.slider_initial_pose.rotate(theta)
        new_slider_target = self.slider_target_pose.rotate(theta)

        # NOTE: Pusher poses are already relative to slider frame, not world frame
        return PlanarPushingStartAndGoal(
            new_slider_init,
            new_slider_target,
            self.pusher_initial_pose,
            self.pusher_target_pose,
        )

    def __str__(self) -> str:
        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]
        return "\n".join(field_strings)


@dataclass
class PlanarPlanConfig:
    # TODO: Add initial and target configuration to this config
    start_and_goal: Optional[PlanarPushingStartAndGoal] = None
    num_knot_points_contact: int = 4
    num_knot_points_non_collision: int = 2
    time_in_contact: float = 2  # TODO: remove, no time
    time_non_collision: float = 0.5  # TODO: remove, there is no time
    continuity_on_pusher_velocity: bool = (
        False  # TODO: Move this into a NonCollisionConfig
    )
    allow_teleportation: bool = False
    use_eq_elimination: bool = False  # TODO: Remove
    use_entry_and_exit_subgraphs: bool = True
    no_cycles: bool = False  # TODO: remove, not used
    workspace: Optional[PlanarPushingWorkspace] = None
    dynamics_config: SliderPusherSystemConfig = field(
        default_factory=lambda: SliderPusherSystemConfig()
    )
    use_band_sparsity: bool = True
    # TODO(bernhardpg): Refactor these cost terms into a struct
    non_collision_cost: NonCollisionCost = field(
        default_factory=lambda: NonCollisionCost()
    )
    contact_config: ContactConfig = field(default_factory=lambda: ContactConfig())
    use_approx_exponential_map: bool = False

    @property
    def dt_contact(self) -> float:
        return self.time_in_contact / self.num_knot_points_contact

    @property
    def dt_non_collision(self) -> float:
        return self.time_non_collision / self.num_knot_points_non_collision

    @property
    def slider_geometry(self) -> CollisionGeometry:
        return self.dynamics_config.slider.geometry

    @property
    def pusher_radius(self) -> float:
        return self.dynamics_config.pusher_radius

    def __str__(self) -> str:
        field_strings = [
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        ]
        return "\n".join(field_strings)
