import argparse
import time
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.common.value import Value
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Cylinder as DrakeCylinder
from pydrake.geometry import (
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    MakePhongIllustrationProperties,
    SceneGraph,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix, eq, ge, le
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    MixedIntegerBranchAndBound,
    MosekSolver,
    Solve,
    SolverOptions,
)
from pydrake.systems.all import (
    ConnectPlanarSceneGraphVisualizer,
    PlanarSceneGraphVisualizer,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
)
from planning_through_contact.geometry.polyhedron import PolyhedronFormulator
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.two_d.equilateral_polytope_2d import (
    EquilateralPolytope2d,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpVariableArray,
)
from planning_through_contact.visualize.analysis import (
    create_quasistatic_pushing_analysis,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

T = TypeVar("T")


def face_name(face_idx: float) -> str:
    return f"face_{face_idx}"


def non_collision_name(face_idx: float) -> str:
    return f"non_collision_{face_idx}"


def forward_differences(
    vars: List[NpExpressionArray], dt: float
) -> List[NpExpressionArray]:
    # TODO: It is cleaner to implement this using a forward diff matrix, but as a first step I do this the simplest way
    forward_diffs = [
        (var_next - var_curr) / dt for var_curr, var_next in zip(vars[0:-1], vars[1:])
    ]
    return forward_diffs


def make_so3(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Takes a SO(2) rotation matrix and returns a rotation matrix in SO(3), where the original matrix
    is treated as a rotation about the z-axis.
    """
    R_in_SO3 = np.eye(3)
    R_in_SO3[0:2, 0:2] = R
    return R_in_SO3


def interpolate_so2_using_slerp(
    Rs: List[npt.NDArray[np.float64]],
    start_time: float,
    end_time: float,
    dt: float,
) -> List[npt.NDArray[np.float64]]:
    """
    Assumes evenly spaced knot points R_matrices.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    Rs_in_SO3 = [make_so3(R) for R in Rs]
    knot_point_times = np.linspace(start_time, end_time, len(Rs))
    quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)

    traj_times = np.arange(start_time, end_time, dt)
    R_traj_in_SO2 = [
        quat_slerp_traj.orientation(t).rotation()[0:2, 0:2] for t in traj_times
    ]

    return R_traj_in_SO2


def interpolate_w_first_order_hold(
    values: npt.NDArray[np.float64],  # (NUM_SAMPLES, NUM_DIMS)
    start_time: float,
    end_time: float,
    dt: float,
) -> npt.NDArray[np.float64]:  # (NUM_POINTS, NUM_DIMS)
    """
    Assumes evenly spaced knot points.

    @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
    """

    knot_point_times = np.linspace(start_time, end_time, len(values))

    # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
    first_order_hold = PiecewisePolynomial.FirstOrderHold(knot_point_times, values.T)
    traj_times = np.arange(start_time, end_time, dt)
    traj = np.hstack(
        [first_order_hold.value(t) for t in traj_times]
    ).T  # transpose to match format in rest of project

    return traj


class ModeVarsResult(NamedTuple):
    cos_ths: npt.NDArray[np.float64]
    sin_ths: npt.NDArray[np.float64]
    p_WBs: npt.NDArray[np.float64]
    p_c_Bs: npt.NDArray[np.float64]
    f_c_Bs: npt.NDArray[np.float64]
    time_in_mode: float

    @property
    def R_WBs(self) -> List[npt.NDArray[np.float64]]:
        Rs = [
            np.array([[cos, -sin], [sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]
        return Rs

    def _rotate_to_world(
        self, vecs_B: npt.NDArray[np.float64]  # (2, num_knot_points)
    ) -> npt.NDArray[np.float64]:
        vecs_W = np.hstack(
            [np.expand_dims(R.dot(v), 1) for R, v in zip(self.R_WBs, vecs_B.T)]
        )
        return vecs_W  # (2, num_knot_points)

    @property
    def p_c_Ws(self) -> npt.NDArray[np.float64]:
        return self.p_WBs + self._rotate_to_world(self.p_c_Bs)

    @property
    def f_c_Ws(self) -> npt.NDArray[np.float64]:
        return self.f_c_Bs

    # Need to handle R_traj as a special case due to List[(2x2)] structure
    def get_R_traj(
        self, dt: float, interpolate: bool = False
    ) -> List[npt.NDArray[np.float64]]:
        if interpolate:
            return interpolate_so2_using_slerp(self.R_WBs, 0, self.time_in_mode, dt)
        else:
            return self.R_WBs

    def _get_traj(
        self,
        knot_points: npt.NDArray[np.float64],
        dt: float,
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:
        if interpolate:
            return interpolate_w_first_order_hold(
                knot_points.T, 0, self.time_in_mode, dt
            )
        else:
            return knot_points.T

    def get_p_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_WBs, dt, interpolate)

    def get_p_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_c_Ws, dt, interpolate)

    def get_f_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.f_c_Ws, dt, interpolate)


# TODO: should probably have a better name
class ModeVars(NamedTuple):
    cos_ths: NpVariableArray  # (1, num_knot_points)
    sin_ths: NpVariableArray  # (1, num_knot_points)
    p_WBs: NpVariableArray  # (2, num_knot_points)
    p_c_Bs: NpExpressionArray  # (2, num_knot_points)
    f_c_Bs: NpExpressionArray  # (2, num_knot_points)
    time_in_mode: float

    def eval_result(self, result: MathematicalProgramResult) -> ModeVarsResult:
        cos_th_vals = result.GetSolution(self.cos_ths)
        sin_th_vals = result.GetSolution(self.sin_ths)
        p_WB_vals = result.GetSolution(self.p_WBs)

        # When result.GetSolution() is called on an expression, it returns an expression, not float
        eval_expr_on_vector = np.vectorize(
            lambda x: x.Evaluate() if isinstance(x, sym.Expression) else x
        )

        p_c_B_vals = eval_expr_on_vector(result.GetSolution(self.p_c_Bs))
        f_c_B_vals = eval_expr_on_vector(result.GetSolution(self.f_c_Bs))

        return ModeVarsResult(
            cos_th_vals,
            sin_th_vals,
            p_WB_vals,
            p_c_B_vals,
            f_c_B_vals,
            self.time_in_mode,
        )


@dataclass
class DynamicsConfig:
    tau_max: float
    f_max: float
    friction_coeff_table_slider: float
    friction_coeff_slider_pusher: float


class PlanarPushingContactMode:
    def __init__(
        self,
        slider: RigidBody,
        contact_face_idx: int,
        dynamics_config: DynamicsConfig,
        num_knot_points: int = 4,
        end_time: float = 3,
        th_initial: Optional[float] = None,
        th_target: Optional[float] = None,
        pos_initial: Optional[npt.NDArray[np.float64]] = None,
        pos_target: Optional[npt.NDArray[np.float64]] = None,
        use_dynamic_constraints: bool = True,
        use_angular_dynamic_constraints: bool = True,
    ):
        self.name = face_name(contact_face_idx)

        self.num_knot_points = num_knot_points
        self.slider = slider
        self.time_in_mode = end_time

        A = np.diag(
            [
                1 / dynamics_config.f_max**2,
                1 / dynamics_config.f_max**2,
                1 / dynamics_config.tau_max**2,
            ]
        )  # Ellipsoidal Limit surface approximation

        dt = end_time / num_knot_points
        self.dt = dt

        prog = MathematicalProgram()

        contact_face = PolytopeContactLocation(
            pos=ContactLocation.FACE, idx=contact_face_idx
        )

        # Contact positions
        self.lams = prog.NewContinuousVariables(num_knot_points, "lam")
        for lam in self.lams:
            prog.AddLinearConstraint(lam >= 0)
            prog.AddLinearConstraint(lam <= 1)

        self.pv1, self.pv2 = self.slider.geometry.get_proximate_vertices_from_location(
            contact_face
        )
        p_c_Bs = [lam * self.pv1 + (1 - lam) * self.pv2 for lam in self.lams]

        # Contact forces
        self.normal_forces = prog.NewContinuousVariables(num_knot_points, "c_n")
        self.friction_forces = prog.NewContinuousVariables(num_knot_points, "c_f")
        (
            self.normal_vec,
            self.tangent_vec,
        ) = self.slider.geometry.get_norm_and_tang_vecs_from_location(contact_face)
        f_c_Bs = [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(self.normal_forces, self.friction_forces)
        ]

        # Rotations
        self.cos_ths = prog.NewContinuousVariables(num_knot_points, "cos_th")
        self.sin_ths = prog.NewContinuousVariables(num_knot_points, "sin_th")
        R_WBs = [
            np.array([[cos, sin], [-sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]

        # Box position relative to world frame
        self.p_WB_xs = prog.NewContinuousVariables(num_knot_points, "p_WB_x")
        self.p_WB_ys = prog.NewContinuousVariables(num_knot_points, "p_WB_y")
        p_WBs = [np.array([x, y]) for x, y in zip(self.p_WB_xs, self.p_WB_ys)]

        # Compute velocities
        v_WBs = forward_differences(p_WBs, dt)
        cos_th_dots = forward_differences(self.cos_ths, dt)
        sin_th_dots = forward_differences(self.sin_ths, dt)
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
        ]
        v_c_Bs = forward_differences(p_c_Bs, dt)

        # In 2D, omega_z = theta_dot will be at position (0,1) in R_dot * R'
        omega_WBs = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(R_WBs, R_WB_dots)]

        # SO(2) constraints
        for c, s in zip(self.cos_ths, self.sin_ths):
            prog.AddConstraint(c**2 + s**2 == 1)

        # # Friction cone constraints
        for c_n in self.normal_forces:
            prog.AddLinearConstraint(c_n >= 0)
        mu = dynamics_config.friction_coeff_slider_pusher
        for c_n, c_f in zip(self.normal_forces, self.friction_forces):
            prog.AddLinearConstraint(c_f <= mu * c_n)
            prog.AddLinearConstraint(c_f >= -mu * c_n)

        # Quasi-static dynamics
        for k in range(num_knot_points - 1):
            v_WB = v_WBs[k]
            omega_WB = omega_WBs[k]
            f_c_B = (f_c_Bs[k] + f_c_Bs[k + 1]) / 2
            p_c_B = (p_c_Bs[k] + p_c_Bs[k + 1]) / 2

            R = np.zeros((3, 3), dtype="O")
            R[2, 2] = 1
            R[0:2, 0:2] = R_WBs[
                k
            ]  # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"

            # Contact torques
            tau_c_B = cross_2d(p_c_B, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench_B = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions

            wrench_W = R.dot(wrench_B)

            # quasi-static dynamics in world frame
            if use_dynamic_constraints:
                quasi_static_dynamic_constraint = eq(x_dot, A.dot(wrench_W))
                linear_vel_constraints = quasi_static_dynamic_constraint[:2]
                for c in linear_vel_constraints:
                    prog.AddConstraint(c)

                # quasi-static dynamics in body frame
                quasi_static_dynamic_constraint = eq(R.T.dot(x_dot), A.dot(wrench_B))
                linear_vel_constraints = quasi_static_dynamic_constraint[:2]
                for c in linear_vel_constraints:
                    prog.AddConstraint(c)

                # Add angular velocity constraints from Lynch et al 1992 equation (11):
                # c = dynamics_config.tau_max / dynamics_config.f_max
                # ang_vel_constraint = omega_WB == (1 / c**2) * cross_2d(p_c_B, v_WB)
                # prog.AddConstraint(ang_vel_constraint)

                s = prog.NewContinuousVariables(1, "s")[0]
                prog.AddConstraint(s == omega_WB)

                t = prog.NewContinuousVariables(1, "t")[0]
                prog.AddConstraint(t == cross_2d(p_c_B, v_WB))  # type: ignore

                prog.AddConstraint(s * t >= 0)

        # Ensure sticking on the contact point
        for v_c_B in v_c_Bs:
            prog.AddLinearConstraint(
                eq(v_c_B, 0)
            )  # no velocity on contact points in body frame

        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in v_WBs])
        sq_angular_vels = sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(cos_th_dots, sin_th_dots)
            ]
        )
        prog.AddQuadraticCost(sq_linear_vels)
        prog.AddQuadraticCost(sq_angular_vels)

        def create_R(th: float) -> npt.NDArray[np.float64]:
            return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        # Initial conditions (only first and last vertex will have this)
        if th_initial is not None:
            R_WB_I = create_R(th_initial)
            prog.AddLinearConstraint(eq(R_WBs[0], R_WB_I))
        if th_target is not None:
            R_WB_T = create_R(th_target)
            prog.AddLinearConstraint(eq(R_WBs[-1], R_WB_T))
        if pos_initial is not None:
            prog.AddLinearConstraint(eq(p_WBs[0], pos_initial.flatten()))
        if pos_target is not None:
            prog.AddLinearConstraint(eq(p_WBs[-1], pos_target.flatten()))

        start = time.time()
        # print("Starting to create SDP relaxation...")
        self.relaxed_prog = MakeSemidefiniteRelaxation(prog)
        end = time.time()
        # print(
        #     f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds"
        # )

        self.num_variables = 7 * num_knot_points + 1  # TODO: 7 is hardcoded, fix this

    def get_spectrahedron(self) -> opt.Spectrahedron:
        # Variables should be the stacked columns of the lower
        # triangular part of the symmetric matrix from relaxed_prog, see
        # https://robotlocomotion.github.io/doxygen_cxx/classdrake_1_1solvers_1_1_mathematical_program.html#a8f718351922bc149cb6e7fa6d82288a5
        spectrahedron = opt.Spectrahedron(self.relaxed_prog)
        return spectrahedron

    def get_vars_from_gcs_vertex(
        self, gcs_vertex: opt.GraphOfConvexSets.Vertex
    ) -> ModeVars:
        x = gcs_vertex.x()[0 : self.num_variables + 1]

        lam = x[0 : self.num_knot_points]
        normal_forces = x[self.num_knot_points : 2 * self.num_knot_points]
        friction_forces = x[2 * self.num_knot_points : 3 * self.num_knot_points]
        cos_ths = x[3 * self.num_knot_points : 4 * self.num_knot_points]
        sin_ths = x[4 * self.num_knot_points : 5 * self.num_knot_points]
        p_WB_xs = x[5 * self.num_knot_points : 6 * self.num_knot_points]
        p_WB_ys = x[6 * self.num_knot_points : 7 * self.num_knot_points]

        p_WBs = np.hstack(
            [np.expand_dims(np.array([x, y]), 1) for x, y in zip(p_WB_xs, p_WB_ys)]
        )

        f_c_Bs = np.hstack(
            [
                c_n * self.normal_vec + c_f * self.tangent_vec
                for c_n, c_f in zip(normal_forces, friction_forces)
            ]
        )

        p_c_Bs = np.hstack([l * self.pv1 + (1 - l) * self.pv2 for l in lam])

        return ModeVars(
            cos_ths,
            sin_ths,
            p_WBs,
            p_c_Bs,
            f_c_Bs,
            self.time_in_mode,
        )

    def eval_result(self, result):
        lam_vals = result.GetSolution(self.lams)
        normal_forces_vals = result.GetSolution(self.normal_forces)
        friction_forces_vals = result.GetSolution(self.friction_forces)
        cos_th_vals = result.GetSolution(self.cos_ths)
        sin_th_vals = result.GetSolution(self.sin_ths)
        p_WB_xs_vals = result.GetSolution(self.p_WB_xs)
        p_WB_ys_vals = result.GetSolution(self.p_WB_ys)

        p_WB_vals = np.hstack(
            [
                np.expand_dims(np.array([x, y]), 1)
                for x, y in zip(p_WB_xs_vals, p_WB_ys_vals)
            ]
        )

        f_c_B_vals = np.hstack(
            [
                c_n * self.normal_vec + c_f * self.tangent_vec
                for c_n, c_f in zip(normal_forces_vals, friction_forces_vals)
            ]
        )

        p_c_B_vals = np.hstack([l * self.pv1 + (1 - l) * self.pv2 for l in lam_vals])

        return ModeVarsResult(
            cos_th_vals,
            sin_th_vals,
            p_WB_vals,
            p_c_B_vals,
            f_c_B_vals,
            self.time_in_mode,
        )

    def solve(self):
        print("Solving...")
        start = time.time()
        result = Solve(self.relaxed_prog)
        end = time.time()
        print(f"Solved in {end - start} seconds")
        assert result.is_success()
        print("Success!")

        lam_vals = result.GetSolution(self.lams)
        normal_forces_vals = result.GetSolution(self.normal_forces)
        friction_forces_vals = result.GetSolution(self.friction_forces)
        cos_th_vals = result.GetSolution(self.cos_ths)
        sin_th_vals = result.GetSolution(self.sin_ths)
        p_WB_xs_vals = result.GetSolution(self.p_WB_xs)
        p_WB_ys_vals = result.GetSolution(self.p_WB_ys)

        p_WB_vals = np.hstack(
            [
                np.expand_dims(np.array([x, y]), 1)
                for x, y in zip(p_WB_xs_vals, p_WB_ys_vals)
            ]
        )

        f_c_B_vals = np.hstack(
            [
                c_n * self.normal_vec + c_f * self.tangent_vec
                for c_n, c_f in zip(normal_forces_vals, friction_forces_vals)
            ]
        )

        p_c_B_vals = np.hstack([l * self.pv1 + (1 - l) * self.pv2 for l in lam_vals])

        return ModeVarsResult(
            cos_th_vals,
            sin_th_vals,
            p_WB_vals,
            p_c_B_vals,
            f_c_B_vals,
            self.time_in_mode,
        )


class NonCollisionMode:
    def __init__(
        self,
        slider: RigidBody,
        name: str,
        non_collision_face_idx: int,
        num_knot_points: int = 3,
        end_time: float = 3,
    ):
        self.num_knot_points = num_knot_points
        self.name = name
        self.time_in_mode = end_time

        # We separate between these two to be able to account for radius,
        # but here we don't care about pusher radius
        faces = slider.geometry.get_planes_for_collision_free_region(
            non_collision_face_idx
        ) + slider.geometry.get_contact_planes(non_collision_face_idx)

        prog = MathematicalProgram()
        # Finger location
        p_BF_xs = prog.NewContinuousVariables(num_knot_points, "p_BF_x")
        p_BF_ys = prog.NewContinuousVariables(num_knot_points, "p_BF_y")
        self.p_BFs = np.hstack(
            [np.expand_dims(np.array([x, y]), 1) for x, y in zip(p_BF_xs, p_BF_ys)]
        )  # (2, num_knot_points)

        self.constraints = []
        for k in range(num_knot_points):
            p_BF = self.p_BFs[:, k]

            for face in faces:
                # TODO: There is a sign error somewhere! b has the wrong sign
                dist_to_face = face.a.T.dot(p_BF) - face.b
                self.constraints.append(ge(dist_to_face, 0))
                prog.AddLinearConstraint(ge(dist_to_face, 0))

        p_WB_x = prog.NewContinuousVariables(1, "p_WB_x").item()
        p_WB_y = prog.NewContinuousVariables(1, "p_WB_y").item()
        self.p_WB = np.expand_dims(np.array([p_WB_x, p_WB_y]), 1)
        self.cos_th = prog.NewContinuousVariables(1, "cos_th").item()
        self.sin_th = prog.NewContinuousVariables(1, "sin_th").item()

        self.vars = np.concatenate(
            [
                self.p_BFs.flatten(order="F"),  # [x1, y1, x2, y2, ...]
                self.p_WB.flatten(),
                [self.cos_th, self.sin_th],
            ]
        )

    def get_polyhedron(self) -> opt.HPolyhedron:
        poly = PolyhedronFormulator(self.constraints).formulate_polyhedron(
            self.vars, make_bounded=False
        )
        return poly

    def get_vars_from_gcs_vertex(
        self, gcs_vertex: opt.GraphOfConvexSets.Vertex
    ) -> ModeVars:
        x = gcs_vertex.x()
        NUM_DIMS = 2
        p_BFs = x[: NUM_DIMS * self.num_knot_points].reshape(
            (NUM_DIMS, self.num_knot_points), order="F"
        )
        p_WB = np.expand_dims(
            x[NUM_DIMS * self.num_knot_points : NUM_DIMS * self.num_knot_points + 2], 1
        )
        cos_th = x[NUM_DIMS * self.num_knot_points + 2]
        sin_th = x[NUM_DIMS * self.num_knot_points + 3]

        # Repeat the variables as we only have one decision variable
        p_WBs = p_WB.repeat(self.num_knot_points, axis=1)
        cos_ths = np.array([cos_th] * self.num_knot_points)
        sin_ths = np.array([sin_th] * self.num_knot_points)

        # TODO for now we just set non-existent forces to zero
        f_c_Bs = np.array(
            [
                [sym.Expression(0)] * self.num_knot_points,
                [sym.Expression(0)] * self.num_knot_points,
            ]
        )

        return ModeVars(cos_ths, sin_ths, p_WBs, p_BFs, f_c_Bs, self.time_in_mode)


def _create_obj_config(
    pos: npt.NDArray[np.float64], th: float
) -> npt.NDArray[np.float64]:
    """
    Concatenates unactauted slider config for source and target vertex
    """
    obj_pose = np.concatenate([pos.flatten(), [np.cos(th), np.sin(th)]])
    return obj_pose


# TODO should be a static method for PlanarPushingContactMode or something
def add_source_or_target_edge(
    vertex: opt.GraphOfConvexSets.Vertex,
    point_vertex: opt.GraphOfConvexSets.Vertex,
    vertex_mode: PlanarPushingContactMode,
    obj_config: npt.NDArray[np.float64],
    gcs: opt.GraphOfConvexSets,
    source_or_target: Literal["source", "target"],
) -> opt.GraphOfConvexSets.Edge:
    vars = vertex_mode.get_vars_from_gcs_vertex(vertex)
    if source_or_target == "source":
        edge = gcs.AddEdge(point_vertex, vertex)
        pos_vars = vars.p_WBs[:, 0]
        cos_var = vars.cos_ths[0]
        sin_var = vars.sin_ths[0]
    else:  # target
        edge = gcs.AddEdge(vertex, point_vertex)
        pos_vars = vars.p_WBs[:, -1]
        cos_var = vars.cos_ths[-1]
        sin_var = vars.sin_ths[-1]

    continuity_constraints = eq(pos_vars.flatten(), obj_config[0:2])
    for c in continuity_constraints.flatten():
        edge.AddConstraint(c)

    continuity_constraint = cos_var == obj_config[2]
    edge.AddConstraint(continuity_constraint)

    continuity_constraint = sin_var == obj_config[3]
    edge.AddConstraint(continuity_constraint)

    return edge


def _find_path_to_target(
    edges: List[opt.GraphOfConvexSets.Edge],
    target: opt.GraphOfConvexSets.Vertex,
    u: opt.GraphOfConvexSets.Vertex,
) -> List[opt.GraphOfConvexSets.Vertex]:
    current_edge = next(e for e in edges if e.u() == u)
    v = current_edge.v()
    target_reached = v == target
    if target_reached:
        return [u] + [v]
    else:
        return [u] + _find_path_to_target(edges, target, v)


# TODO should be a static method for PlanarPushingContactMode or something
def add_edge_with_continuity_constraint(
    u_vertex: opt.GraphOfConvexSets.Vertex,
    v_vertex: opt.GraphOfConvexSets.Vertex,
    u_mode: Union[PlanarPushingContactMode, NonCollisionMode],
    v_mode: Union[PlanarPushingContactMode, NonCollisionMode],
    gcs: opt.GraphOfConvexSets,
) -> opt.GraphOfConvexSets.Edge:
    edge = gcs.AddEdge(u_vertex, v_vertex)

    u_vars = u_mode.get_vars_from_gcs_vertex(u_vertex)
    v_vars = v_mode.get_vars_from_gcs_vertex(v_vertex)

    cont_constraints = create_continuity_constraints(u_vars, v_vars)
    for c in cont_constraints:
        edge.AddConstraint(c)

    return edge


def create_continuity_constraints(
    incoming_vars: ModeVars, outgoing_vars: ModeVars
) -> NpFormulaArray:
    constraints = []
    constraints.append(
        eq(incoming_vars.p_c_Bs[:, -1], outgoing_vars.p_c_Bs[:, 0]).flatten()
    )
    constraints.append(
        eq(incoming_vars.p_WBs[:, -1], outgoing_vars.p_WBs[:, 0]).flatten()
    )
    constraints.append(
        [
            incoming_vars.cos_ths[-1] == outgoing_vars.cos_ths[0],
            incoming_vars.sin_ths[-1] == outgoing_vars.sin_ths[0],
        ]
    )
    return np.concatenate(constraints)


@dataclass
class GraphChain:
    start_contact_idx: int
    end_contact_idx: int
    non_collision_chains: List[List[int]]
    no_cycles: bool = True

    @classmethod
    def from_contact_connection(
        cls,
        incoming: int,
        outgoing: int,
        paths: Dict[Tuple[int, int], List[List[int]]],
        no_cycles: bool,
    ) -> "GraphChain":
        non_collision_idx_on_chain = paths[(incoming, outgoing)]  # type: ignore

        return cls(incoming, outgoing, non_collision_idx_on_chain, no_cycles)

    def create_contact_modes(
        self, slider: RigidBody, num_knot_points: int, time_in_each_mode: float
    ) -> None:
        self.non_collision_modes = [
            [
                NonCollisionMode(
                    slider,
                    f"from_{self.start_contact_idx}_to_{self.end_contact_idx}_at_{idx}",
                    idx,
                    num_knot_points,
                    time_in_each_mode,
                )
                for idx in chain
            ]
            for chain in self.non_collision_chains
        ]

    def create_edges(
        self,
        start_contact_vertex: opt.GraphOfConvexSets.Vertex,
        end_contact_vertex: opt.GraphOfConvexSets.Vertex,
        start_contact_mode: PlanarPushingContactMode,
        end_contact_mode: PlanarPushingContactMode,
        gcs: opt.GraphOfConvexSets,
    ) -> None:
        self.non_collision_vertices = [
            [
                gcs.AddVertex(mode.get_polyhedron(), name=mode.name)
                for mode in mode_chain
            ]
            for mode_chain in self.non_collision_modes
        ]

        for chain, modes, vertices in zip(
            self.non_collision_chains,
            self.non_collision_modes,
            self.non_collision_vertices,
        ):
            # Connect contact mode to first position mode
            add_edge_with_continuity_constraint(
                start_contact_vertex,
                vertices[0],
                start_contact_mode,
                modes[0],
                gcs,
            )

            if not self.no_cycles:
                add_edge_with_continuity_constraint(
                    vertices[0],
                    start_contact_vertex,
                    modes[0],
                    start_contact_mode,
                    gcs,
                )

            # Connect last position mode to last contact mode
            add_edge_with_continuity_constraint(
                vertices[-1],
                end_contact_vertex,
                modes[-1],
                end_contact_mode,
                gcs,
            )

            if not self.no_cycles:
                add_edge_with_continuity_constraint(
                    end_contact_vertex,
                    vertices[-1],
                    end_contact_mode,
                    modes[-1],
                    gcs,
                )

            for i in range(len(chain) - 1):
                curr_vertex = vertices[i]
                curr_mode = modes[i]
                next_vertex = vertices[i + 1]
                next_mode = modes[i + 1]
                add_edge_with_continuity_constraint(
                    curr_vertex, next_vertex, curr_mode, next_mode, gcs
                )
                add_edge_with_continuity_constraint(
                    next_vertex, curr_vertex, next_mode, curr_mode, gcs
                )

    def add_costs(self) -> None:
        for modes, vertices in zip(
            self.non_collision_modes, self.non_collision_vertices
        ):
            for mode, vertex in zip(modes, vertices):
                vars = mode.get_vars_from_gcs_vertex(vertex)
                diffs = vars.p_c_Bs[:, 1:] - vars.p_c_Bs[:, :-1]  # type: ignore
                squared_eucl_dist = sum([d.T.dot(d) for d in diffs.T])
                vertex.AddCost(squared_eucl_dist)
                # TODO: Remove
                # vars = mode.get_vars_from_gcs_vertex(vertex)
                # cost = sum([p.T.dot(p) for p in vars.p_c_Bs.T])
                # vertex.AddCost(cost)

    def get_all_non_collision_vertices(
        self,
    ) -> List[List[opt.GraphOfConvexSets.Vertex]]:
        assert len(self.non_collision_vertices) > 0
        return self.non_collision_vertices

    def get_all_non_collision_modes(self) -> List[List[NonCollisionMode]]:
        assert len(self.non_collision_modes) > 0
        return self.non_collision_modes


@dataclass
class StartEndSpecs:
    th_initial: float
    th_target: float
    pos_initial: npt.NDArray[np.float64]
    pos_target: npt.NDArray[np.float64]


@dataclass
class PlanarTraj:
    R_traj: List[npt.NDArray[np.float64]]
    com_traj: npt.NDArray[np.float64]
    contact_pos_traj: npt.NDArray[np.float64]
    force_traj: npt.NDArray[np.float64]
    dt: float

    def __len__(self) -> int:
        return len(self.R_traj)

    def get_idx_for_t(self, t: float) -> int:
        return min(int(np.floor(t / self.dt)), len(self) - 1)

    def end_time(self) -> float:
        return len(self) * self.dt


class TrajGeometry(LeafSystem):
    def __init__(
        self,
        traj: PlanarTraj,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        scene_graph: SceneGraph,
    ) -> None:
        super().__init__()

        self.traj = traj

        self.slider_geometry = slider_geometry
        self.pusher_radius = pusher_radius

        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()
        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider")
        )
        BOX_COLOR = COLORS["aquamarine4"]
        DEFAULT_HEIGHT = 0.3
        if isinstance(slider_geometry, Box2d):
            box_geometry_id = scene_graph.RegisterGeometry(
                self.source_id,
                self.slider_frame_id,
                GeometryInstance(
                    RigidTransform.Identity(),
                    DrakeBox(
                        slider_geometry.width, slider_geometry.height, DEFAULT_HEIGHT
                    ),
                    "slider",
                ),
            )
            scene_graph.AssignRole(
                self.source_id,
                box_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse()),
            )
        elif isinstance(slider_geometry, TPusher2d) or isinstance(
            slider_geometry, ArbitraryShape2D
        ):
            boxes, transforms = slider_geometry.get_as_boxes(DEFAULT_HEIGHT / 2)
            box_geometry_ids = [
                scene_graph.RegisterGeometry(
                    self.source_id,
                    self.slider_frame_id,
                    GeometryInstance(
                        transform,
                        DrakeBox(box.width, box.height, DEFAULT_HEIGHT),
                        f"box_{idx}",
                    ),
                )
                for idx, (box, transform) in enumerate(zip(boxes, transforms))
            ]
            for box_geometry_id in box_geometry_ids:
                scene_graph.AssignRole(
                    self.source_id,
                    box_geometry_id,
                    MakePhongIllustrationProperties(BOX_COLOR.diffuse()),
                )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher"),
        )
        CYLINDER_HEIGHT = 0.3
        self.pusher_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.pusher_frame_id,
            GeometryInstance(
                RigidTransform(
                    RotationMatrix.Identity(), np.array([0, 0, CYLINDER_HEIGHT / 2])  # type: ignore
                ),
                DrakeCylinder(pusher_radius, CYLINDER_HEIGHT),
                "pusher",
            ),
        )
        FINGER_COLOR = COLORS["firebrick3"]
        scene_graph.AssignRole(
            self.source_id,
            self.pusher_geometry_id,
            MakePhongIllustrationProperties(FINGER_COLOR.diffuse()),
        )

        # TODO: Shows table
        # TABLE_COLOR = COLORS["bisque3"]
        # TABLE_HEIGHT = 0.1
        # table_geometry_id = scene_graph.RegisterAnchoredGeometry(
        #     self.source_id,
        #     GeometryInstance(
        #         RigidTransform(
        #             RotationMatrix.Identity(), np.array([0, 0, -TABLE_HEIGHT / 2])  # type: ignore
        #         ),
        #         DrakeBox(1.0, 1.0, TABLE_HEIGHT),
        #         "table",
        #     ),
        # )
        # scene_graph.AssignRole(
        #     self.source_id,
        #     table_geometry_id,
        #     MakePhongIllustrationProperties(TABLE_COLOR.diffuse()),
        # )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        traj: PlanarTraj,
        slider_geometry: CollisionGeometry,
        scene_graph: SceneGraph,
        pusher_radius: float,
        name: str = "traj_geometry ",
    ) -> "TrajGeometry":
        traj_geometry = builder.AddNamedSystem(
            name,
            cls(
                traj,
                slider_geometry,
                pusher_radius,
                scene_graph,
            ),
        )
        builder.Connect(
            traj_geometry.get_output_port(),
            scene_graph.get_source_pose_port(traj_geometry.source_id),
        )
        return traj_geometry

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        t = context.get_time()
        idx = self.traj.get_idx_for_t(t)

        R_WB = self.traj.R_traj[idx]
        p_WB = self.traj.com_traj[idx, :]
        p_WP = self.traj.contact_pos_traj[idx, :]
        # TODO force

        R_WB_so3 = np.eye(3)
        R_WB_so3[:2, :2] = R_WB
        p_x = p_WB[0]
        p_y = p_WB[1]

        slider_pose = RigidTransform(
            RotationMatrix(R_WB_so3), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.pusher_frame_id, value=pusher_pose)  # type: ignore


def visualize_planar_pushing_trajectory_drake(
    traj: PlanarTraj,
    slider_geometry: CollisionGeometry,
    show: bool = False,
    save: bool = False,
):
    builder = DiagramBuilder()
    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    traj_geometry = TrajGeometry.add_to_builder(
        builder,
        traj,
        slider_geometry,
        scene_graph,
        0.015,
    )

    max_x_y = max(
        max(np.abs(traj.contact_pos_traj.flatten())),
        max(np.abs(traj.com_traj.flatten())),
    )
    max_geometry_dist = max(
        [max(np.abs(v.flatten())) for v in slider_geometry.vertices]
    )
    max_x_y_and_geometry = max_x_y + max_geometry_dist
    LIM = max_x_y_and_geometry * 1.1

    def connect_planar_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph
    ) -> PlanarSceneGraphVisualizer:
        T_VW = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        visualizer = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            T_VW=T_VW,
            xlim=[-LIM, LIM],
            ylim=[-LIM, LIM],
            show=show,
        )
        return visualizer

    visualizer = connect_planar_visualizer(builder, scene_graph)

    diagram = builder.Build()
    diagram.set_name("diagram")

    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png("diagram_viz.png")  # type: ignore

    # Create the simulator, and simulate for 10 seconds.
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)

    visualizer.start_recording()  # type: ignore

    simulator.Initialize()
    # simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(traj.end_time())

    visualizer.stop_recording()  # type: ignore
    ani = visualizer.get_recording_as_animation()  # type: ignore

    if save:
        # Playback the recording and save the output.
        ani.save("test.mp4", fps=30)

    return ani


def plan_planar_pushing(
    start_end_specs: StartEndSpecs,
    slider: RigidBody,
    dynamics_config: DynamicsConfig,
    max_rounded_paths: int,
    print_output: bool = False,
    visualize_old: bool = False,
    no_cycles: bool = True,
    use_angular_dynamics: bool = True,
    use_dynamics: bool = True,
    save_ani: bool = False,
) -> opt.GraphOfConvexSets:
    faces_to_consider = list(range(slider.geometry.num_collision_free_regions))

    # Build the graph
    source_connections = faces_to_consider
    target_connections = faces_to_consider

    face_connections = list(combinations(faces_to_consider, 2))

    NUM = len(faces_to_consider)

    def create_path(i, j):
        if i >= NUM or j >= NUM:
            raise ValueError(f"i and j must be below {NUM}")
        if i < j:
            return [n for n in range(i, j + 1)]
        else:
            return [n % NUM for n in range(i, j + 1 + NUM)]

    contact_map = slider.geometry.get_collision_free_region_for_loc_idx

    paths = {
        (i, j): (
            create_path(contact_map(i), contact_map(j)),
            list(reversed(create_path(contact_map(j), contact_map(i)))),
        )
        for (i, j) in face_connections
    }

    num_knot_points = 4
    time_in_contact = 2
    time_moving = 0.5

    initial_config = _create_obj_config(
        start_end_specs.pos_initial, start_end_specs.th_initial
    )
    target_config = _create_obj_config(
        start_end_specs.pos_target, start_end_specs.th_target
    )

    source_point = opt.Point(initial_config)
    target_point = opt.Point(target_config)

    gcs = opt.GraphOfConvexSets()
    source_vertex = gcs.AddVertex(source_point, name="source")
    target_vertex = gcs.AddVertex(target_point, name="target")

    contact_modes = [
        PlanarPushingContactMode(
            slider,
            face_idx,
            dynamics_config,
            num_knot_points=num_knot_points,
            end_time=time_in_contact,
            use_dynamic_constraints=use_dynamics,
            use_angular_dynamic_constraints=use_angular_dynamics,
        )
        for face_idx in faces_to_consider
    ]
    spectrahedrons = [mode.get_spectrahedron() for mode in contact_modes]
    contact_vertices = [
        gcs.AddVertex(s, name=str(m.name))
        for s, m in zip(spectrahedrons, contact_modes)
    ]

    # Add costs
    for mode, vertex in zip(contact_modes, contact_vertices):
        prog = mode.relaxed_prog
        for cost in prog.linear_costs():
            vars = vertex.x()[prog.FindDecisionVariableIndices(cost.variables())]
            a = cost.evaluator().a()
            vertex.AddCost(a.T.dot(vars))

    for idx in source_connections:
        vertex = contact_vertices[idx]
        mode = contact_modes[idx]

        add_source_or_target_edge(
            vertex,
            source_vertex,
            mode,
            initial_config,
            gcs,
            source_or_target="source",
        )

    for idx in target_connections:
        vertex = contact_vertices[idx]
        mode = contact_modes[idx]

        add_source_or_target_edge(
            vertex,
            target_vertex,
            mode,
            target_config,
            gcs,
            source_or_target="target",
        )

    num_knot_points_for_non_collision = 2

    chains = [
        GraphChain.from_contact_connection(incoming_idx, outgoing_idx, paths, no_cycles)
        for incoming_idx, outgoing_idx in face_connections
    ]
    for chain in chains:
        chain.create_contact_modes(
            slider, num_knot_points_for_non_collision, time_moving
        )

    for chain in chains:
        incoming_vertex = contact_vertices[chain.start_contact_idx]
        outgoing_vertex = contact_vertices[chain.end_contact_idx]
        incoming_mode = contact_modes[chain.start_contact_idx]
        outgoing_mode = contact_modes[chain.end_contact_idx]
        chain.create_edges(
            incoming_vertex, outgoing_vertex, incoming_mode, outgoing_mode, gcs
        )
        chain.add_costs()

    # Collect all modes and vertices in one big lookup table for trajectory retrieval
    # NOTE: There is really no reason to keep the contact_vertices and contact_modes as dicts, but
    # this will not be fixed now.

    graphviz = gcs.GetGraphvizString()
    data = pydot.graph_from_dot_data(graphviz)[0]
    data.write_svg("graph.svg")

    all_vertices = {v.id(): v for v in contact_vertices}
    all_modes = {v.id(): m for v, m in zip(contact_vertices, contact_modes)}

    for chain in chains:
        mode_chain = chain.get_all_non_collision_modes()
        vertex_chain = chain.get_all_non_collision_vertices()

        for modes, vertices in zip(mode_chain, vertex_chain):
            for mode, vertex in zip(modes, vertices):
                v_id = vertex.id()
                all_modes[v_id] = mode  # type: ignore
                all_vertices[v_id] = vertex

    # Make sure we have all the vertices (except for source and target)
    assert len(all_vertices.items()) == len(gcs.Vertices()) - 2

    options = opt.GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.solver_options = SolverOptions()
    if print_output:
        options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # options.solver_options.SetOption(
    #     MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3
    # )
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
    # options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
    # options.solver_options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
    # options.solver_options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)
    if options.convex_relaxation is True:
        options.preprocessing = True  # TODO Do I need to deal with this?
        options.max_rounded_paths = max_rounded_paths
    import time

    start = time.time()
    result = gcs.SolveShortestPath(source_vertex, target_vertex, options)
    elapsed_time = time.time() - start

    assert result.is_success()
    print("Success!")

    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [result.GetSolution(p) for p in flow_variables]

    flow_values_per_edge = {
        (e.u().name(), e.v().name(), e.u().id(), e.v().id()): flow
        for e, flow in zip(gcs.Edges(), flow_results)
    }

    active_edges = [
        edge for edge, flow in zip(gcs.Edges(), flow_results) if flow >= 0.55
    ]

    full_path = _find_path_to_target(active_edges, target_vertex, source_vertex)
    vertex_ids_on_path = [
        v.id() for v in full_path if v.name() not in ["source", "target"]
    ]

    vertices_on_path = [all_vertices[id] for id in vertex_ids_on_path]

    vertex_names_on_path = [v.name() for v in vertices_on_path]

    def print_path(names):
        print("Path:")
        print(", ".join(names))

    print_path(vertex_names_on_path)

    modes_on_path = [all_modes[id] for id in vertex_ids_on_path]

    mode_vars_on_path = [
        mode.get_vars_from_gcs_vertex(vertex)
        for mode, vertex in zip(modes_on_path, vertices_on_path)
    ]
    vals = [mode.eval_result(result) for mode in mode_vars_on_path]

    dets = [np.linalg.det(R) for val in vals for R in val.R_WBs]

    def print_floats(float_list):
        # Define the number of decimals you want to display
        decimals = 2

        # Format each float and join them into a single line
        formatted_line = ", ".join([f"{x:.{decimals}f}" for x in float_list])
        print(formatted_line)

    print("Determinants:")
    print_floats(dets)
    assert np.allclose(dets, 1, atol=1e-3)  # type: ignore

    DT = 0.5
    interpolate = False
    R_traj = sum(
        [val.get_R_traj(DT, interpolate=interpolate) for val in vals],
        [],
    )
    com_traj = np.vstack(
        [val.get_p_WB_traj(DT, interpolate=interpolate) for val in vals]
    )
    force_traj = np.vstack(
        [val.get_f_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )
    contact_pos_traj = np.vstack(
        [val.get_p_c_W_traj(DT, interpolate=interpolate) for val in vals]
    )

    traj_length = len(R_traj)

    traj = PlanarTraj(R_traj, com_traj, contact_pos_traj, force_traj, DT)

    compute_violation = False
    if compute_violation:
        # NOTE: SHOULD BE MOVED!
        # compute quasi-static dynamic violation
        def _cross_2d(v1, v2):
            return (
                v1[0] * v2[1] - v1[1] * v2[0]
            )  # copied because the other one assumes the result is a np array, here it is just a scalar. clean up!

        quasi_static_violation = []
        for k in range(traj_length - 1):
            v_WB = (com_traj[k + 1] - com_traj[k]) / DT
            R_dot = (R_traj[k + 1] - R_traj[k]) / DT
            R = R_traj[k]
            omega_WB = R_dot.dot(R.T)[1, 0]
            f_c_B = force_traj[k]
            p_c_B = contact_pos_traj[k]
            com = com_traj[k]

            # Contact torques
            tau_c_B = _cross_2d(p_c_B - com, f_c_B)

            x_dot = np.concatenate([v_WB, [omega_WB]])
            wrench = np.concatenate(
                [f_c_B.flatten(), [tau_c_B]]
            )  # NOTE: Should fix not nice vector dimensions

            R_padded = np.zeros((3, 3))
            R_padded[2, 2] = 1
            R_padded[0:2, 0:2] = R
            violation = x_dot - R_padded.dot(A).dot(wrench)
            quasi_static_violation.append(violation)

        quasi_static_violation = np.vstack(quasi_static_violation)
        create_quasistatic_pushing_analysis(quasi_static_violation, num_knot_points)
        plt.show()

    if not visualize_old:
        ani = visualize_planar_pushing_trajectory_drake(
            traj, slider.geometry, save=save_ani
        )
        return gcs, ani
    else:
        CONTACT_COLOR = COLORS["dodgerblue4"]
        GRAVITY_COLOR = COLORS["blueviolet"]
        BOX_COLOR = COLORS["aquamarine4"]
        TABLE_COLOR = COLORS["bisque3"]
        FINGER_COLOR = COLORS["firebrick3"]
        TARGET_COLOR = COLORS["firebrick1"]

        flattened_rotation = np.vstack([R.flatten() for R in R_traj])
        box_viz = VisualizationPolygon2d.from_trajs(
            com_traj,
            flattened_rotation,
            slider.geometry,
            BOX_COLOR,
        )

        # NOTE: I don't really need the entire trajectory here, but leave for now
        target_viz = VisualizationPolygon2d.from_trajs(
            com_traj,
            flattened_rotation,
            slider.geometry,
            TARGET_COLOR,
        )

        com_points_viz = VisualizationPoint2d(com_traj, GRAVITY_COLOR)  # type: ignore
        contact_point_viz = VisualizationPoint2d(contact_pos_traj, FINGER_COLOR)  # type: ignore
        contact_point_viz.change_radius(0.01)
        contact_force_viz = VisualizationForce2d(contact_pos_traj, CONTACT_COLOR, force_traj)  # type: ignore

        viz = Visualizer2d()
        FRAMES_PER_SEC = 1 / DT
        viz.visualize(
            [contact_point_viz],
            [contact_force_viz],
            [box_viz],
            FRAMES_PER_SEC,
            target_viz,
        )

        return gcs, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        help="Which experiment to run",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    experiment_number = args.exp

    mass = 0.1
    slider = RigidBody("t_pusher", TPusher2d(), mass)
    # slider = RigidBody("box", Box2d(width=0.3, height=0.3), mass)

    f_max = 0.5 * 9.81 * slider.mass
    dynamics_config = DynamicsConfig(
        friction_coeff_table_slider=0.5,
        friction_coeff_slider_pusher=0.5,
        f_max=f_max,
        tau_max=f_max * 0.5,
    )

    if experiment_number == 0:  # not tight
        th_initial = 0
        th_target = 0.5
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[0.2, 0.2]])

    elif experiment_number == 1:  # tight
        th_initial = 0
        th_target = 0.5
        pos_initial = np.array([[0.2, 0.1]])
        pos_target = np.array([[-0.2, 0.2]])

    elif experiment_number == 2:  # tight
        th_initial = 0.0
        th_target = -0.6
        pos_initial = np.array([[0.4, 0.4]])
        pos_target = np.array([[0.1, 0.3]])

    elif experiment_number == 3:  # tight
        th_initial = 0
        th_target = -0.68
        pos_initial = np.array([[0.0, 0.5]])
        pos_target = np.array([[0.2, 0.2]])

    elif experiment_number == 4:  # tight
        th_initial = 0
        th_target = 0.4
        pos_initial = np.array([[0.2, 0.2]])
        pos_target = np.array([[-0.18, 0.5]])

    start_end_specs = StartEndSpecs(th_initial, th_target, pos_initial, pos_target)

    plan_planar_pushing(
        start_end_specs,
        slider,
        dynamics_config,
        max_rounded_paths=1,
        print_output=True,
        visualize_old=False,
        no_cycles=True,
        save_ani=True,
    )
