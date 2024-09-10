from dataclasses import dataclass, fields
from functools import cached_property
from logging import Logger
from pathlib import Path
from time import time
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from pydrake.common.containers import EqualToDict
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, SnoptSolver
from pydrake.symbolic import Expression, Variable, Variables
from pydrake.systems.controllers import DiscreteTimeLinearQuadraticRegulator
from tqdm import tqdm

from planning_through_contact.convex_relaxation.sdp import (
    EqualityEliminationType,
    ImpliedConstraintsType,
    compute_optimality_gap_pct,
    find_solution,
    get_gaussian_from_sdp_relaxation_solution,
    solve_sdp_relaxation,
    visualize_sparsity,
)
from planning_through_contact.tools.math import null_space_basis_qr_pivot
from planning_through_contact.tools.script_utils import (
    YamlMixin,
    default_script_setup,
    get_current_git_commit,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.colors import BROWN2, BURLYWOOD3, BURLYWOOD4

IntegratorType = Literal["forward_euler", "backward_euler"]


@dataclass
class LinearComplementaritySystem:
    """
    Continuous time:
    ẋ(t) = f(x(t), u(t), λ(t)) = Ax(t) + Bu(t) + Dλ(t) + d

    or

    Discrete time:
    x[k+1] = f(x[k], u[k], λ[k]) = Ax[k] + Bu[k] + Dλ[k] + d

    and

    0 ≤ λ ⊥ Ex + Fλ + Hu + c ≥ 0
    """

    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    D: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    F: npt.NDArray[np.float64]
    H: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    discrete_or_continuous: Literal["discrete", "continuous"]

    @property
    def num_states(self) -> int:
        return self.A.shape[1]

    @property
    def num_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def num_forces(self) -> int:
        return self.D.shape[1]

    def get_f(self, x: np.ndarray, u: np.ndarray, λ: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u + self.D @ λ + self.d

    def get_complementarity_rhs(
        self, x: np.ndarray, u: np.ndarray, λ: np.ndarray
    ) -> np.ndarray:
        return self.E @ x + self.F @ λ + self.H @ u + self.c

    def discretize(
        self, integrator: IntegratorType, T_s: float
    ) -> "LinearComplementaritySystem":
        if self.discrete_or_continuous != "continuous":
            raise RuntimeError("System is already discrete!")

        # Discretize the continuous time system (self.sys)
        if integrator == "forward_euler":
            A_discrete = np.eye(self.num_states) + T_s * self.A
            B_discrete = T_s * self.B
            D_discrete = T_s * self.D
            d_discrete = T_s * self.d
        else:
            raise NotImplementedError(
                f"Integrator type {integrator} is not implemented."
            )

        return LinearComplementaritySystem(
            A_discrete,
            B_discrete,
            D_discrete,
            d_discrete,
            self.E,
            self.F,
            self.H,
            self.c,
            "discrete",
        )


class CartPoleWithWalls(LinearComplementaritySystem):
    """
    Cart-pole with walls system from the paper:

    A. Aydinoglu, P. Sieg, V. M. Preciado, and M. Posa,
    “Stabilization of Complementarity Systems via
    Contact-Aware Controllers.” 2021

    x_1 = cart position
    x_2 = pole position
    x_3 = cart velocity
    x_4 = pole velocity

    u_1 = force applied to cart

    λ_1 = contact force from right wall
    λ_2 = contact force from left wall

    """

    def __init__(self):
        # fmt: off
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 3.51, 0, 0],
            [0, 22.2, 0, 0]
        ])
        
        B = np.array([
            [0],
            [0],
            [1.02],
            [1.7]
        ])
        
        D = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [4.7619, -4.7619]
        ])
        
        d = np.zeros((4,))

        E = np.array([
            [-1, 0.6, 0, 0],
            [1, -0.6, 0, 0]
        ])
        
        F = np.array([
            [0.0014, 0],
            [0, 0.0014]
        ])
        
        H = np.zeros((2,1))
        
        c = np.array([0.35, 0.35])
        # fmt: on

        super().__init__(A, B, D, d, E, F, H, c, "continuous")

        self.distance_to_walls = 0.35
        self.pole_length = 0.6
        self.cart_mass = 0.978
        self.pole_mass = 0.35

    def get_linearized_pole_top_position(self, x: float, θ: float) -> float:
        """
        Returns the linearized pole position:
        x - l sin θ ≈ x - l θ

        @param x is the cart position.
        @param θ is the pole angle.
        """
        return x - self.pole_length * θ

    def get_linearized_distance_to_wall(
        self, x: float, θ: float, wall: Literal["right", "left"]
    ) -> float:
        """
        Returns the linearized distance to the wall, where the pole position is taken as
        x - l sin θ ≈ x - l θ

        @param x is the cart position.
        @param θ is the pole angle.
        """
        if wall == "right":
            return self.distance_to_walls - self.get_linearized_pole_top_position(x, θ)
        else:  # left:
            return self.distance_to_walls + self.get_linearized_pole_top_position(x, θ)


@dataclass
class CartPoleWithWallsTrajectory:
    cart_position: npt.NDArray[np.float64]
    pole_angle: npt.NDArray[np.float64]
    cart_velocity: npt.NDArray[np.float64]
    pole_velocity: npt.NDArray[np.float64]
    applied_force: npt.NDArray[np.float64]
    right_contact_force: npt.NDArray[np.float64]
    left_contact_force: npt.NDArray[np.float64]
    sys: CartPoleWithWalls
    T_s: float

    def __post_init__(self):
        states = "cart_position", "pole_angle", "cart_velocity", "pole_velocity"

        for field in fields(self):
            if field.name in ("sys", "T_s"):
                continue
            if field.name in states:
                if len(getattr(self, field.name)) != self.state_length:
                    raise ValueError(
                        f"All state trajectories must be of length {self.state_length}"
                    )
            else:
                if len(getattr(self, field.name)) != self.input_length:
                    raise ValueError(
                        f"All input/force trajectories must be of length {self.input_length}"
                    )

    @cached_property
    def linearized_pole_top_position(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.sys.get_linearized_pole_top_position(x, θ)
                for x, θ in zip(self.cart_position, self.pole_angle)
            ]
        )

    @classmethod
    def from_state_input_forces(
        cls,
        state: npt.NDArray[np.float64],
        input: npt.NDArray[np.float64],
        forces: npt.NDArray[np.float64],
        sys: CartPoleWithWalls,
        T_s: float,
    ) -> "CartPoleWithWallsTrajectory":
        if state.shape[1] != 4:
            raise ValueError("States must have shape (N, 4)")

        if input.shape[1] != 1:
            raise ValueError("Inputs must have shape (N, 1)")

        if forces.shape[1] != 2:
            raise ValueError("Forces must have shape (N, 2)")

        cart_pos = state[:, 0]
        pole_pos = state[:, 1]
        cart_vel = state[:, 2]
        pole_vel = state[:, 3]
        applied_force = input.flatten()
        right_wall_force = forces[:, 0]
        left_wall_force = forces[:, 1]

        return cls(
            cart_pos,
            pole_pos,
            cart_vel,
            pole_vel,
            applied_force,
            right_wall_force,
            left_wall_force,
            sys,
            T_s,
        )

    @property
    def state_length(self) -> int:
        return len(self.cart_position)

    @property
    def input_length(self) -> int:
        return len(self.cart_position) - 1

    def plot(self, filepath: Path | None = None) -> None:
        # Create a figure and subplots.
        fig, axs = plt.subplots(4, 2, figsize=(12, 6), sharex=True)

        # Ensure axs is a list of Axes.
        if isinstance(axs, Axes):
            axs = np.array([[axs]])
        elif len(axs.shape) == 1:
            axs = axs[:, np.newaxis]

        # Plot each trajectory and adjust y-axis limits.
        def plot_with_dynamic_limits(ax, data, label, color):
            ax.plot(data, label=label, color=color)
            ax.scatter(range(len(data)), data, color=color, s=10)  # Plot keypoints.
            ax.set_title(label)

            data_range = data.max() - data.min()
            TOL = 1e-1
            if data_range < TOL:  # Adjust this threshold as needed.
                ax.set_ylim(data.min() - TOL, data.max() + TOL)

        plot_with_dynamic_limits(
            axs[0][0], self.cart_position, "Cart Position [m]", "blue"
        )
        plot_with_dynamic_limits(
            axs[1][0], self.cart_velocity, "Cart Velocity [m/s]", "blue"
        )
        plot_with_dynamic_limits(
            axs[2][0], self.pole_angle * 180 / np.pi, "Pole Angle [deg]", "orange"
        )

        plot_with_dynamic_limits(
            axs[3][0],
            self.pole_velocity * 180 / np.pi,
            "Pole (angular) Velocity [deg/s]",
            "orange",
        )

        plot_with_dynamic_limits(
            axs[0][1], self.applied_force, "Applied force [N]", "green"
        )
        plot_with_dynamic_limits(
            axs[1][1], self.left_contact_force, "(Left) Contact force [N]", "red"
        )
        plot_with_dynamic_limits(
            axs[2][1], self.right_contact_force, "(Right) Contact force [N]", "red"
        )

        plot_with_dynamic_limits(
            axs[3][1],
            self.linearized_pole_top_position,
            "(Linearized) Pole top position [m]",
            "purple",
        )
        # Plot wall positions
        axs[3][1].axhline(y=self.sys.distance_to_walls, color="grey", linestyle="--")
        axs[3][1].axhline(y=-self.sys.distance_to_walls, color="grey", linestyle="--")

        # Set the x-axis label for the last subplots.
        axs[3][0].set_xlabel("Time Step")
        axs[3][1].set_xlabel("Time Step")

        # Display the plots
        plt.tight_layout()

        if filepath is not None:
            fig.savefig(filepath)
        else:
            plt.show()

    def animate(self, output_file: Path | None = None) -> None:
        # Set up the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")

        # Initialize the cart and pole
        cart_width = 0.4
        cart_height = 0.3
        pole_length = self.sys.pole_length

        cart = Rectangle(
            (0, 0), cart_width, cart_height, fc=BROWN2.diffuse(), ec="black"
        )
        ax.add_patch(cart)
        pole = Line2D(
            [],
            [],
            lw=3,
            color=BURLYWOOD3.diffuse(),  # type: ignore
            marker="o",
            markersize=9,
            markerfacecolor=BURLYWOOD4.diffuse(),  # type: ignore
            markeredgecolor="black",
        )
        ax.add_line(pole)

        # Initialize walls
        wall_distance = self.sys.distance_to_walls
        wall_height = 0.4
        wall_width = 0.1

        left_wall = Rectangle(
            (-wall_distance - wall_width / 2, 0.3),
            wall_width,
            wall_height,
            fc="grey",
            ec="black",
        )
        right_wall = Rectangle(
            (wall_distance + wall_width / 2, 0.3),
            wall_width,
            wall_height,
            fc="grey",
            ec="black",
        )
        ax.add_patch(left_wall)
        ax.add_patch(right_wall)

        # Initialize force arrows using FancyArrowPatch
        FORCE_SCALE = 1

        def create_force_patch(color):
            return FancyArrowPatch(
                (0, 0),
                (0, 0),
                arrowstyle="->",
                color=color,
                mutation_scale=8,
                linewidth=2,
                zorder=10,
            )

        applied_force_arrow = create_force_patch("blue")
        left_contact_force_arrow = create_force_patch("blue")
        right_contact_force_arrow = create_force_patch("blue")

        ax.add_patch(applied_force_arrow)
        ax.add_patch(left_contact_force_arrow)
        ax.add_patch(right_contact_force_arrow)

        def init():
            cart.set_xy((-cart_width / 2, -cart_height / 2))
            pole.set_data([], [])
            applied_force_arrow.set_positions((0, 0), (0, 0))
            left_contact_force_arrow.set_positions((0, 0), (0, 0))
            right_contact_force_arrow.set_positions((0, 0), (0, 0))
            return (
                cart,
                pole,
                applied_force_arrow,
                left_contact_force_arrow,
                right_contact_force_arrow,
            )

        def update(frame):
            x = float(self.cart_position[frame])
            theta = float(self.pole_angle[frame])
            applied_force = float(self.applied_force[frame])
            left_contact_force = float(self.left_contact_force[frame])
            right_contact_force = float(self.right_contact_force[frame])

            cart.set_xy((x - cart_width / 2, -cart_height / 2))

            rod_x = x + pole_length * np.sin(theta)
            rod_y = pole_length * np.cos(theta)

            pole_x = [x, rod_x]
            pole_y = [0, rod_y]
            pole.set_data(pole_x, pole_y)

            # Define a threshold for plotting forces
            FORCE_THRESHOLD = 1e-3

            # Helper function to update force arrows conditionally
            def update_horizontal_force_arrow(arrow, start_pos, force):
                if abs(force) >= FORCE_THRESHOLD:
                    arrow.set_positions(
                        start_pos,
                        (start_pos[0] + force * FORCE_SCALE, start_pos[1]),
                    )
                else:
                    arrow.set_positions((0, 0), (0, 0))

            # Update force arrows
            update_horizontal_force_arrow(applied_force_arrow, (x, 0), applied_force)
            update_horizontal_force_arrow(
                left_contact_force_arrow,
                (rod_x, rod_y),
                left_contact_force,
            )
            update_horizontal_force_arrow(
                right_contact_force_arrow,
                (rod_x, rod_y),
                -right_contact_force,  # this force is along the negative x-direction.
            )

            return (
                cart,
                pole,
                applied_force_arrow,
                left_contact_force_arrow,
                right_contact_force_arrow,
            )

        interval_ms = int(self.T_s * 1000)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=self.input_length,
            init_func=init,
            blit=True,
            interval=interval_ms,
        )

        if output_file is not None:
            ani.save(output_file, writer="ffmpeg")
        else:
            plt.show()


# TODO: Move to unit test
def test_cart_pole_w_walls():
    sys = CartPoleWithWalls()
    x = np.zeros((sys.num_states,))
    u = np.zeros((sys.num_inputs,))
    λ = np.zeros((sys.num_forces,))
    assert isinstance(sys.get_f(x, u, λ), type(np.array([])))
    assert sys.get_f(x, u, λ).shape == (sys.num_states,)

    assert isinstance(sys.get_complementarity_rhs(x, u, λ), type(np.array([])))
    assert sys.get_complementarity_rhs(x, u, λ).shape == (sys.num_forces,)

    # Test distance functions
    # (Upright positions)
    distance = sys.get_linearized_distance_to_wall(0.35, 0, wall="right")
    assert distance == 0

    # (Upright positions)
    distance = sys.get_linearized_distance_to_wall(-0.35, 0, wall="left")
    assert distance == 0

    # (Some angle)
    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="left"
    ) > sys.get_linearized_distance_to_wall(0, 0.1, wall="left")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="left"
    ) < sys.get_linearized_distance_to_wall(0, -0.1, wall="left")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="right"
    ) < sys.get_linearized_distance_to_wall(0, 0.1, wall="right")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="right"
    ) > sys.get_linearized_distance_to_wall(0, -0.1, wall="right")


@dataclass
class TrajectoryOptimizationParameters:
    N: int
    T_s: float
    Q: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    Q_N: npt.NDArray[np.float64] | None = None
    integrator: IntegratorType = "forward_euler"


# TODO: Delete this, this was just to test something quick
class CartPoleMechanicalEliminationTrajopt:
    def __init__(
        self,
        continuous_sys: LinearComplementaritySystem,
        params: TrajectoryOptimizationParameters,
        x0: npt.NDArray[np.float64],
    ):
        sys = continuous_sys.discretize(params.integrator, params.T_s)

        prog = MathematicalProgram()

        pos_idxs = [0, 1]
        num_pos = len(pos_idxs)

        ps = prog.NewContinuousVariables(params.N + 1, num_pos, "p")

        # p0, v0 = np.split(x0, [2])
        # ps = np.vstack([p0, ps])  # add the initial condition

        vs = np.vstack(
            [(p_next - p) / params.T_s for p, p_next in zip(ps[:-1], ps[1:])]
        )
        vs = np.vstack(
            [vs, np.zeros_like(vs[0], dtype=float)]
        )  # add an extra zero at the end
        # vs = np.vstack([v0, vs])  # add the initial condition

        xs = np.hstack([ps, vs])
        # xs = np.vstack([x0, xs])  # add the initial condition
        us = prog.NewContinuousVariables(params.N, sys.num_inputs, "u")
        λs = prog.NewContinuousVariables(params.N, sys.num_forces, "λ")

        self.sys = sys
        self.params = params
        self.qcqp = prog
        self.xs = xs
        self.ps = ps
        self.vs = vs
        self.us = us
        self.λs = λs

        initial_condition = eq(xs[0], x0)
        for c in initial_condition:
            prog.AddLinearConstraint(c)

        # TODO: remove (code for not eliminating x0 manually)
        # xs = qcqp.NewContinuousVariables(params.N + 1, sys.num_states, "x")
        # initial_condition = eq(vs[0], v0)
        # for c in initial_condition:
        #     prog.AddLinearConstraint(c)
        # breakpoint()

        vel_idxs = [2, 3]
        # Dynamics
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            v_next = vs[k + 1]

            A_a = sys.A[vel_idxs, :]
            B_a = sys.B[vel_idxs, :]
            D_a = sys.D[vel_idxs, :]
            # TODO: Remember affine term later (For cart pole it is zero)

            dynamics = eq(v_next, A_a @ x + B_a @ u + D_a @ λ)

            prog.AddLinearConstraint(dynamics)

        # # Dynamics
        # for k in range(params.N):
        #     x, u, λ = xs[k], us[k], λs[k]
        #     x_next = xs[k + 1]
        #
        #     f = sys.get_f(x, u, λ)
        #     dynamics = eq(x_next, f)
        #     const = prog.AddLinearConstraint(dynamics)

        # RHS nonnegativity of complementarity constraint
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]

            rhs = sys.get_complementarity_rhs(x, u, λ)
            prog.AddLinearConstraint(ge(rhs, 0))

        # LHs nonnegativity of complementarity constraint
        for k in range(params.N):
            λ = λs[k]
            prog.AddLinearConstraint(ge(λ, 0))

        # Complementarity constraint (element-wise)
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            rhs = sys.get_complementarity_rhs(x, u, λ)

            elementwise_product = λ * rhs
            for p in elementwise_product:
                prog.AddQuadraticConstraint(p, 0, 0)  # p == 0

        # Input limits
        # TODO

        # State limits
        # TODO

        # Cost
        for k in range(params.N):
            x, u = xs[k], us[k]
            prog.AddQuadraticCost(x.T @ params.Q @ x)

            u = u.reshape((-1, 1))  # handle the case where u.shape = (1,)
            prog.AddQuadraticCost((u.T @ params.R @ u).item())

        # Terminal cost
        if params.Q_N is not None:
            Q_N = params.Q_N
        else:
            _, S = DiscreteTimeLinearQuadraticRegulator(
                sys.A, sys.B, params.Q, params.R
            )
            Q_N = S  # use the infinite-horizon optimal cost-to-go as the terminal cost

        prog.AddQuadraticCost(xs[params.N].T @ Q_N @ xs[params.N])

    def evaluate_state_input_forces(
        self, result: MathematicalProgramResult
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        # We always evaluate xs as an expression, because it is a mix of floats (initial condition) and vars
        xs_sol = evaluate_np_expressions_array(self.xs, result)

        if type(self.us[0]) is Expression:
            us_sol = evaluate_np_expressions_array(self.us, result)
        else:
            us_sol = result.GetSolution(self.us).reshape((-1, 1))

        if type(self.λs[0]) is Expression:
            λs_sol = evaluate_np_expressions_array(self.λs, result)
        else:
            λs_sol = result.GetSolution(self.λs)

        return xs_sol, us_sol, λs_sol

    def get_state_input_forces_from_decision_var_values(
        self,
        vals: npt.NDArray[np.float64],
        equality_elimination_method: EqualityEliminationType | None = None,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Given an array of decision variable values, return the corresponding xs, us, λs.
        """
        if not len(vals) == len(self.qcqp.decision_variables()):
            raise RuntimeError(
                f"Number of provided values does not match number of decision\
                variables: #vals = {len(vals)}, #decision_variables = {len(self.qcqp.decision_variables())}"
            )

        if equality_elimination_method is not None:
            var_values = {
                var: val for var, val in zip(self.qcqp.decision_variables(), vals)
            }

        var_to_idx = EqualToDict(
            {var: idx for idx, var in enumerate(self.qcqp.decision_variables())}
        )

        def get_val_or_keep(var_or_val_or_expr: float | Variable | Expression) -> float:
            if type(var_or_val_or_expr) == float:
                return var_or_val_or_expr
            elif type(var_or_val_or_expr) == Variable:
                val_idx = var_to_idx[var_or_val_or_expr]
                return float(vals[val_idx])
            elif type(var_or_val_or_expr) == Expression:
                val = var_or_val_or_expr.Evaluate(var_values)  # type: ignore
                return val
            else:
                breakpoint()
                raise RuntimeError("Wrong type")

        get_vals = np.vectorize(get_val_or_keep)
        return get_vals(self.xs), get_vals(self.us), get_vals(self.λs)

    def get_vars_at_time_step(self, k: int) -> np.ndarray:
        assert k <= self.params.N
        # First entry of self.xs is just x0 (which is a constant)
        return np.concatenate([self.xs[k + 1], self.us[k], self.λs[k]])

    def get_variable_groups(self) -> list[Variables]:
        variable_groups = [
            Variables(
                np.concatenate(
                    [self.get_vars_at_time_step(k), self.get_vars_at_time_step(k + 1)]
                )
            )
            for k in range(self.params.N - 1)
        ]
        return variable_groups

    @staticmethod
    def compute_nullspace_basis(
        params: TrajectoryOptimizationParameters,
        sys: LinearComplementaritySystem,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute the null space basis for equality constraints arising
        from the dynamics constraints:

             x[k+1] = Ax[k] + Bu[k] + Dλ[k] + d
           ⟹ x[k+1] = Ax[k] + Bu[k] + Dλ[k] - x[k+1] = -d
           ⟹ [A B D -I] [x[k]  ] = -d
                        [u[k]  ]
                        [λ[k]  ]
                        [x[k+1]]

        which we rename as:

            A_eq y = b_eq

        This function computes F and ŷ such that y = Fz + ŷ.

        """
        I = np.eye(sys.num_states)
        A_eq = np.block([sys.A, sys.B, sys.D, -I])
        b_eq = -sys.d

        # y = Fz + ŷ
        F = null_space_basis_qr_pivot(A_eq)

        # visualize_sparsity(F, color=True)

        # TODO: It seems that 0 is always a solution?
        ŷ = find_solution(A_eq, b_eq)

        return F, ŷ

    @staticmethod
    def define_decision_vars_in_latent_space(
        params: TrajectoryOptimizationParameters,
        prog: MathematicalProgram,
        sys: LinearComplementaritySystem,
        nullspace_basis: npt.NDArray[np.float64],
        particular_solution: npt.NDArray[np.float64],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define the decision variables through a latent space variable obtained
        from elimination of equality constraints.
        """

        F, ŷ = nullspace_basis, particular_solution
        N_latent_vars = F.shape[1]
        zs = prog.NewContinuousVariables(params.N, N_latent_vars, "z")
        concatenated_vars = np.vstack([F @ z_k + ŷ for z_k in zs])

        split_sizes = [
            sys.num_states,
            sys.num_inputs,
            sys.num_forces,
        ]
        split_idxs = np.cumsum(split_sizes)
        xs, us, λs, xs_next = np.split(concatenated_vars, split_idxs, axis=1)
        return xs, us, λs, xs_next


class LcsTrajectoryOptimization:
    def __init__(
        self,
        continuous_sys: LinearComplementaritySystem,
        params: TrajectoryOptimizationParameters,
        x0: npt.NDArray[np.float64],
        equality_elimination_method: EqualityEliminationType | None = None,
    ):
        sys = continuous_sys.discretize(params.integrator, params.T_s)

        prog = MathematicalProgram()

        # Define states, inputs, and forces
        if equality_elimination_method is None:
            xs = prog.NewContinuousVariables(params.N, sys.num_states, "x")
            # Add initial state to state vector
            xs = np.vstack([x0, xs])
            us = prog.NewContinuousVariables(params.N, sys.num_inputs, "u")
            λs = prog.NewContinuousVariables(params.N, sys.num_forces, "λ")

            add_dynamics = True

        # TODO: Remove this, it does absolutely nothing more than
        # adding variables which are then later removed.
        elif equality_elimination_method == "blockwise_qr_pivot":
            F, ŷ = self.compute_nullspace_basis(params, sys)
            xs, us, λs, xs_next = self.define_decision_vars_in_latent_space(
                params, prog, sys, F, ŷ
            )
            # Either add initial condition as a constraint or by substitution
            # (uncomment the next line)
            # xs = np.vstack([x0, xs])
            initial_condition = eq(xs[0], x0)
            for c in initial_condition:
                prog.AddLinearEqualityConstraint(c)

            xs = np.vstack([xs, xs_next[-1]])
            for x_next_1, x_next_2 in zip(xs[1:], xs_next[:-1]):
                for c in eq(x_next_1, x_next_2):
                    prog.AddLinearEqualityConstraint(c)

            add_dynamics = True

        elif equality_elimination_method == "shooting":
            us = prog.NewContinuousVariables(params.N, sys.num_inputs, "u")
            λs = prog.NewContinuousVariables(params.N, sys.num_forces, "λ")

            # Compute A̅ and B̅ such that
            # x[k+1] = A̅[k]* x[k] + B̅[k]* u̅[k]
            # where
            # u̅[k] = [u[0], …, u[k-1]],
            # A̅[k] = Aᵏ, and B̅[k] = [Aᵏ⁻¹B, …, AB, B]

            A_bars = [np.linalg.matrix_power(sys.A, k) for k in range(params.N + 1)]

            def _get_next_matrix(
                k: int, M: npt.NDArray[np.float64]
            ) -> npt.NDArray[np.float64]:
                """
                Given a matrix M, returns M̅[k] = [Aᵏ⁻¹M, …, AM, M]
                """
                M̅_blocks = [A̅_k @ M for A̅_k in reversed(A_bars[: k + 1])]
                return np.block(M̅_blocks)

            B_bars = [_get_next_matrix(k, sys.B) for k in range(params.N)]
            D_bars = [_get_next_matrix(k, sys.D) for k in range(params.N)]
            d_bars = [M @ sys.d for M in np.cumsum(A_bars, axis=0)]

            # Dynamics
            xs = np.zeros((params.N + 1, sys.num_states), dtype=object)
            xs[0] = x0
            for k in range(params.N):
                # TODO: It might be worth adding some unit tests here because this is very prone to an index error!
                u_bar, λ_bar = us[: k + 1].flatten(), λs[: k + 1].flatten()
                A_bar = A_bars[k + 1]
                B_bar = B_bars[k]
                D_bar = D_bars[k]
                d_bar = d_bars[k]
                x_next = A_bar @ x0 + B_bar @ u_bar + D_bar @ λ_bar + d_bar

                xs[k + 1] = x_next

            add_dynamics = False

        else:
            raise NotImplementedError("")

        self.add_trajopt_cost_and_constraints(
            params, sys, prog, xs, us, λs, add_dynamics
        )

        self.sys = sys
        self.params = params
        self.qcqp = prog
        self.xs = xs
        self.us = us
        self.λs = λs

        # TODO: remove (code for not eliminating x0 manually)
        # xs = qcqp.NewContinuousVariables(params.N + 1, sys.num_states, "x")
        # initial_condition = eq(xs[0], x0)
        # for c in initial_condition:
        #     qcqp.AddLinearConstraint(c)

    @staticmethod
    def add_trajopt_cost_and_constraints(
        params: TrajectoryOptimizationParameters,
        sys: LinearComplementaritySystem,
        prog: MathematicalProgram,
        xs: np.ndarray,
        us: np.ndarray,
        λs: np.ndarray,
        add_dynamics: bool = True,
    ) -> None:
        """
        Given a MathematicalProgram and the state variables (`xs`), input variables (`us`), and force variables (`λs`),
        this function adds all the trajectory optimization constraints and costs to it according to the provided `sys`
        and `params`.

        The variables are assumed to have shape (N, m) where N is the trajectory optimization horizon length and m is
        the number of variables per time step.
        """

        if add_dynamics:
            for k in range(params.N):
                x, u, λ = xs[k], us[k], λs[k]
                x_next = xs[k + 1]

                f = sys.get_f(x, u, λ)
                dynamics = eq(x_next, f)
                prog.AddLinearConstraint(dynamics)

        # RHS nonnegativity of complementarity constraint
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]

            rhs = sys.get_complementarity_rhs(x, u, λ)
            prog.AddLinearConstraint(ge(rhs, 0))

        # LHs nonnegativity of complementarity constraint
        for k in range(params.N):
            λ = λs[k]
            prog.AddLinearConstraint(ge(λ, 0))

        # Complementarity constraint (element-wise)
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            rhs = sys.get_complementarity_rhs(x, u, λ)

            elementwise_product = λ * rhs
            for p in elementwise_product:
                prog.AddQuadraticConstraint(p, 0, 0)  # p == 0

        # Input limits
        # TODO

        # State limits
        # TODO

        # Cost
        for k in range(params.N):
            x, u = xs[k], us[k]
            cost = prog.AddQuadraticCost(x.T @ params.Q @ x)

            if k > 0:
                WARNING_TRESH = 1e8
                abs_val = np.abs(cost.evaluator().Q()).max()
                if abs_val > WARNING_TRESH:
                    logger.warning(f"Huge value found in Q: {abs_val}")

            u = u.reshape((-1, 1))  # handle the case where u.shape = (1,)
            prog.AddQuadraticCost((u.T @ params.R @ u).item())

        # Terminal cost
        if params.Q_N is not None:
            Q_N = params.Q_N
        else:
            _, S = DiscreteTimeLinearQuadraticRegulator(
                sys.A, sys.B, params.Q, params.R
            )
            Q_N = S  # use the infinite-horizon optimal cost-to-go as the terminal cost

        prog.AddQuadraticCost(xs[params.N].T @ Q_N @ xs[params.N])

    def evaluate_state_input_forces(
        self, result: MathematicalProgramResult
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        # We always evaluate xs as an expression, because it is a mix of floats (initial condition) and vars
        xs_sol = evaluate_np_expressions_array(self.xs, result)

        if type(self.us[0]) is Expression:
            us_sol = evaluate_np_expressions_array(self.us, result)
        else:
            us_sol = result.GetSolution(self.us).reshape((-1, 1))

        if type(self.λs[0]) is Expression:
            λs_sol = evaluate_np_expressions_array(self.λs, result)
        else:
            λs_sol = result.GetSolution(self.λs)

        return xs_sol, us_sol, λs_sol

    def get_state_input_forces_from_decision_var_values(
        self,
        vals: npt.NDArray[np.float64],
        equality_elimination_method: EqualityEliminationType | None = None,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Given an array of decision variable values, return the corresponding xs, us, λs.
        """
        if not len(vals) == len(self.qcqp.decision_variables()):
            raise RuntimeError(
                f"Number of provided values does not match number of decision\
                variables: #vals = {len(vals)}, #decision_variables = {len(self.qcqp.decision_variables())}"
            )

        if equality_elimination_method is not None:
            var_values = {
                var: val for var, val in zip(self.qcqp.decision_variables(), vals)
            }

        var_to_idx = EqualToDict(
            {var: idx for idx, var in enumerate(self.qcqp.decision_variables())}
        )

        def get_val_or_keep(var_or_val_or_expr: float | Variable | Expression) -> float:
            if type(var_or_val_or_expr) == float:
                return var_or_val_or_expr
            elif type(var_or_val_or_expr) == Variable:
                val_idx = var_to_idx[var_or_val_or_expr]
                return float(vals[val_idx])
            elif type(var_or_val_or_expr) == Expression:
                val = var_or_val_or_expr.Evaluate(var_values)  # type: ignore
                return val
            else:
                breakpoint()
                raise RuntimeError("Wrong type")

        get_vals = np.vectorize(get_val_or_keep)
        return get_vals(self.xs), get_vals(self.us), get_vals(self.λs)

    def get_vars_at_time_step(self, k: int) -> np.ndarray:
        assert k <= self.params.N
        # First entry of self.xs is just x0 (which is a constant)
        return np.concatenate([self.xs[k + 1], self.us[k], self.λs[k]])

    def get_variable_groups(self) -> list[Variables]:
        variable_groups = [
            Variables(
                np.concatenate(
                    [self.get_vars_at_time_step(k), self.get_vars_at_time_step(k + 1)]
                )
            )
            for k in range(self.params.N - 1)
        ]
        return variable_groups

    @staticmethod
    def compute_nullspace_basis(
        params: TrajectoryOptimizationParameters,
        sys: LinearComplementaritySystem,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute the null space basis for equality constraints arising
        from the dynamics constraints:

             x[k+1] = Ax[k] + Bu[k] + Dλ[k] + d
           ⟹ x[k+1] = Ax[k] + Bu[k] + Dλ[k] - x[k+1] = -d
           ⟹ [A B D -I] [x[k]  ] = -d
                        [u[k]  ]
                        [λ[k]  ]
                        [x[k+1]]

        which we rename as:

            A_eq y = b_eq

        This function computes F and ŷ such that y = Fz + ŷ.

        """
        I = np.eye(sys.num_states)
        A_eq = np.block([sys.A, sys.B, sys.D, -I])
        b_eq = -sys.d

        # y = Fz + ŷ
        F = null_space_basis_qr_pivot(A_eq)

        # visualize_sparsity(F, color=True)

        # TODO: It seems that 0 is always a solution?
        ŷ = find_solution(A_eq, b_eq)

        return F, ŷ

    @staticmethod
    def define_decision_vars_in_latent_space(
        params: TrajectoryOptimizationParameters,
        prog: MathematicalProgram,
        sys: LinearComplementaritySystem,
        nullspace_basis: npt.NDArray[np.float64],
        particular_solution: npt.NDArray[np.float64],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define the decision variables through a latent space variable obtained
        from elimination of equality constraints.
        """

        F, ŷ = nullspace_basis, particular_solution
        N_latent_vars = F.shape[1]
        zs = prog.NewContinuousVariables(params.N, N_latent_vars, "z")
        concatenated_vars = np.vstack([F @ z_k + ŷ for z_k in zs])

        split_sizes = [
            sys.num_states,
            sys.num_inputs,
            sys.num_forces,
        ]
        split_idxs = np.cumsum(split_sizes)
        xs, us, λs, xs_next = np.split(concatenated_vars, split_idxs, axis=1)
        return xs, us, λs, xs_next


# TODO: Move to unit test
def test_lcs_trajectory_optimization():
    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )

    x0 = np.array([0, 0, 0, 0])

    trajopt = LcsTrajectoryOptimization(sys, params, x0)

    assert trajopt.xs.shape == (params.N + 1, sys.num_states)
    assert trajopt.us.shape == (params.N, sys.num_inputs)
    assert trajopt.λs.shape == (params.N, sys.num_forces)

    # Dynamics
    assert (
        len(trajopt.qcqp.linear_equality_constraints()) == (params.N) * sys.num_states
    )

    # Complementarity LHS and RHS ≥ 0
    assert (
        len(trajopt.qcqp.linear_constraints())
        + len(trajopt.qcqp.bounding_box_constraints())
        == params.N * 2
    )

    # Complementarity constraints
    assert len(trajopt.qcqp.quadratic_constraints()) == params.N * sys.num_forces

    # Running costs (2 per time step) + terminal cost
    assert len(trajopt.qcqp.quadratic_costs()) == 2 * params.N + 1

    for k in range(params.N):
        group = trajopt.get_vars_at_time_step(k)
        assert group.shape == (sys.num_states + sys.num_inputs + sys.num_forces,)
        for var in group:
            assert type(var) == Variable

    groups = trajopt.get_variable_groups()
    assert len(groups) == params.N - 1

    for group in groups:
        assert isinstance(group, Variables)
        assert len(group) == (sys.num_states + sys.num_inputs + sys.num_forces) * 2


# TODO: Move to unit test
def test_lcs_trajopt_with_sparsity_construction():
    sys = CartPoleWithWalls()

    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )

    F, ŷ = LcsTrajectoryOptimization.compute_nullspace_basis(params, sys)
    num_original_vars = 2 * sys.num_states + sys.num_inputs + sys.num_forces
    num_latent_space_vars = F.shape[1]

    # We have num_states number of equality constraints (dynamics) hence we
    # should reduce this many
    assert num_original_vars - num_latent_space_vars == sys.num_states

    assert F.shape[0] == num_original_vars
    assert ŷ.shape[0] == num_original_vars

    xs, us, λs, xs_next = (
        LcsTrajectoryOptimization.define_decision_vars_in_latent_space(
            params, MathematicalProgram(), sys, F, ŷ
        )
    )

    assert xs.shape[0] == params.N
    assert us.shape[0] == params.N
    assert λs.shape[0] == params.N
    assert xs_next.shape[0] == params.N

    assert xs.shape[1] == sys.num_states
    assert us.shape[1] == sys.num_inputs
    assert λs.shape[1] == sys.num_forces
    assert xs_next.shape[1] == sys.num_states


# TODO: Move to unit test
def test_lcs_trajopt_with_sparsity():
    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )
    x0 = np.array([0, 0, 0, 0])

    trajopt = LcsTrajectoryOptimization(
        sys, params, x0, equality_elimination_method="qr_pivot"
    )

    breakpoint()


def test_lcs_get_state_input_forces_from_vals():
    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    trajopt = LcsTrajectoryOptimization(sys, params, x0)
    prog = trajopt.qcqp

    np.random.seed(0)
    vals = np.random.rand(*prog.decision_variables().shape)
    assert isinstance(vals, type(np.array([])))
    xs, us, λs = trajopt.get_state_input_forces_from_decision_var_values(vals)

    assert np.allclose(xs[0, :], x0)
    assert xs.shape == trajopt.xs.shape
    assert us.shape == trajopt.us.shape
    assert λs.shape == trajopt.λs.shape


@dataclass
class RoundingTrial:
    success: bool
    time: float
    cost: float
    result: MathematicalProgramResult

    def __str__(self) -> str:
        return (
            f"success: {self.success}, time: {self.time:.4f} s, cost: {self.cost:.4f}"
        )


def plot_rounding_trials(
    trials: list[RoundingTrial], output_dir: Path | None = None
) -> None:
    num_rounding_trials = len(trials)

    # Extract attributes for plotting
    success_values = [trial.success for trial in trials]
    time_values = [trial.time for trial in trials]
    cost_values = [trial.cost for trial in trials]

    # Find the index of the trial with the lowest cost
    best_trial_index = cost_values.index(min(cost_values))

    # Plotting the attributes
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    # Ensure axs is a list of Axes
    if isinstance(axs, Axes):
        axs = [axs]

    # Plot success values
    axs[0].bar(range(num_rounding_trials), success_values, color="grey")
    axs[0].bar(
        best_trial_index, success_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[0].set_xlabel("Trial Index")
    axs[0].set_ylabel("Success")
    axs[0].set_title("Rounding Trial Success")
    axs[0].set_xticks(range(num_rounding_trials))

    # Plot time values
    axs[1].bar(range(num_rounding_trials), time_values, color="grey")
    axs[1].bar(
        best_trial_index, time_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[1].set_xlabel("Trial Index")
    axs[1].set_ylabel("Time (s)")
    axs[1].set_title("Rounding Trial Time")
    axs[1].set_xticks(range(num_rounding_trials))

    # Plot cost values
    axs[2].bar(range(num_rounding_trials), cost_values, color="grey")
    axs[2].bar(
        best_trial_index, cost_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[2].set_xlabel("Trial Index")
    axs[2].set_ylabel("Cost")
    axs[2].set_title("Rounding Trial Cost")
    axs[2].set_xticks(range(num_rounding_trials))

    plt.tight_layout()

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / "rounding_trials.pdf")


@dataclass
class CartPoleConfig(YamlMixin):
    trajopt_params: TrajectoryOptimizationParameters
    x0: npt.NDArray[np.float64]
    implied_constraints: ImpliedConstraintsType
    equality_elimination_method: EqualityEliminationType | None
    use_trace_cost: float | None
    use_chain_sparsity: bool
    seed: int
    num_rounding_trials: int
    git_commit: str


# TODO: This is just to quickly see if only eliminating some variables maintains tightness.
# If not, this code can be deleted.
def cart_pole_test_mechanical_elimination(
    output_dir: Path, debug: bool, logger: Logger
) -> None:
    sys = CartPoleWithWalls()
    Q = np.diag([10, 100, 1, 10])

    cfg = CartPoleConfig(
        trajopt_params=TrajectoryOptimizationParameters(
            N=20,
            T_s=0.1,
            Q=Q,
            R=np.array([1]),
        ),
        x0=np.array([0.3, 0, 0.10, 0]),
        implied_constraints="weakest",
        equality_elimination_method="qr_pivot",
        use_trace_cost=1e-5,
        use_chain_sparsity=False,
        seed=0,
        num_rounding_trials=5,
        git_commit=get_current_git_commit(),
    )

    cfg.save(output_dir / "config.yaml")

    np.random.seed(cfg.seed)

    logger.info("Building trajopt program...")
    trajopt = CartPoleMechanicalEliminationTrajopt(
        sys,
        cfg.trajopt_params,
        cfg.x0,
    )

    logger.info("Solving SDP relaxation...")
    Y, relaxed_cost, relaxed_result = solve_sdp_relaxation(
        qcqp=trajopt.qcqp,
        trace_cost=cfg.use_trace_cost,
        implied_constraints=cfg.implied_constraints,
        variable_groups=(
            trajopt.get_variable_groups() if cfg.use_chain_sparsity else None
        ),
        print_time=True,
        plot_eigvals=True,
        print_eigvals=True,
        logger=logger,
        output_dir=output_dir,
        equality_elimination_method=cfg.equality_elimination_method,
    )

    # Rounding
    μ, Σ = get_gaussian_from_sdp_relaxation_solution(Y)

    eigvec = np.linalg.eig(Y).eigenvectors[0]
    eigvec = eigvec / eigvec[0]

    # Save "mean"/relaxed trajectory
    relaxed_trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
        *trajopt.get_state_input_forces_from_decision_var_values(
            μ, cfg.equality_elimination_method
        ),
        sys,
        cfg.trajopt_params.T_s,
    )
    relaxed_trajectory.plot(output_dir / "relaxed_trajectory.pdf")
    relaxed_trajectory.animate(output_dir / "relaxed_animation.mp4")

    # Save "mean"/relaxed trajectory
    eigenvector_trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
        *trajopt.get_state_input_forces_from_decision_var_values(
            μ, cfg.equality_elimination_method
        ),
        sys,
        cfg.trajopt_params.T_s,
    )
    eigenvector_trajectory.plot(output_dir / "eigenvector_trajectory.pdf")
    eigenvector_trajectory.animate(output_dir / "eigenvector_animation.mp4")

    initial_guesses = [μ]  # use the mean as an initial guess
    initial_guesses.extend(
        np.random.multivariate_normal(mean=μ, cov=Σ, size=cfg.num_rounding_trials)
    )

    trials = []
    logger.info(f"Rounding {len(initial_guesses)} trials...")
    for initial_guess in tqdm(initial_guesses):
        snopt = SnoptSolver()

        start = time()
        result = snopt.Solve(trajopt.qcqp, initial_guess)  # type: ignore
        end = time()
        rounding_time = end - start

        trial = RoundingTrial(
            result.is_success(), rounding_time, result.get_optimal_cost(), result
        )
        trials.append(trial)

    plot_rounding_trials(trials, output_dir)

    best_trial_idx = np.argmin([trial.cost for trial in trials])
    best_trial = trials[best_trial_idx]

    for idx, trial in enumerate(trials):
        logger.info(
            f"Trial {idx}: {trial}, optimality gap (upper bound): {compute_optimality_gap_pct(trial.cost, relaxed_cost):.3f} %"
        )
        trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
            *trajopt.evaluate_state_input_forces(trial.result),
            sys,
            cfg.trajopt_params.T_s,
        )

        trial_dir = output_dir / f"trial_{idx}"

        if idx == best_trial_idx:
            trial_dir = Path(str(trial_dir) + "_BEST")
        trial_dir.mkdir(exist_ok=True)
        trajectory.plot(trial_dir / "trajectory.pdf")
        trajectory.animate(trial_dir / "animation.mp4")

    logger.info(f"Best trial: {best_trial_idx}")
    logger.info(
        f"Best optimality gap: {compute_optimality_gap_pct(best_trial.cost, relaxed_cost):.4f}%"
    )


def cart_pole_experiment_1(output_dir: Path, debug: bool, logger: Logger) -> None:
    sys = CartPoleWithWalls()
    Q = np.diag([10, 100, 1, 10])

    cfg = CartPoleConfig(
        trajopt_params=TrajectoryOptimizationParameters(
            N=20,
            T_s=0.1,
            Q=Q,
            R=np.array([1]),
        ),
        x0=np.array([0.3, 0, 0.10, 0]),
        implied_constraints="weakest",
        equality_elimination_method="shooting",
        use_trace_cost=1e-5,
        use_chain_sparsity=False,
        seed=0,
        num_rounding_trials=5,
        git_commit=get_current_git_commit(),
    )

    cfg.save(output_dir / "config.yaml")

    np.random.seed(cfg.seed)

    logger.info("Building trajopt program...")
    trajopt = LcsTrajectoryOptimization(
        sys,
        cfg.trajopt_params,
        cfg.x0,
        equality_elimination_method=cfg.equality_elimination_method,
    )

    logger.info("Solving SDP relaxation...")
    Y, relaxed_cost, _ = solve_sdp_relaxation(
        qcqp=trajopt.qcqp,
        trace_cost=cfg.use_trace_cost,
        implied_constraints=cfg.implied_constraints,
        variable_groups=(
            trajopt.get_variable_groups() if cfg.use_chain_sparsity else None
        ),
        print_time=True,
        plot_eigvals=True,
        print_eigvals=True,
        logger=logger,
        output_dir=output_dir,
    )

    # Rounding
    μ, Σ = get_gaussian_from_sdp_relaxation_solution(Y)

    # Save "mean"/relaxed trajectory
    relaxed_trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
        *trajopt.get_state_input_forces_from_decision_var_values(
            μ, cfg.equality_elimination_method
        ),
        sys,
        cfg.trajopt_params.T_s,
    )
    relaxed_trajectory.plot(output_dir / "relaxed_trajectory.pdf")
    relaxed_trajectory.animate(output_dir / "relaxed_animation.mp4")

    initial_guesses = [μ]  # use the mean as an initial guess
    initial_guesses.extend(
        np.random.multivariate_normal(mean=μ, cov=Σ, size=cfg.num_rounding_trials)
    )

    trials = []
    logger.info(f"Rounding {len(initial_guesses)} trials...")
    for initial_guess in tqdm(initial_guesses):
        snopt = SnoptSolver()

        start = time()
        result = snopt.Solve(trajopt.qcqp, initial_guess)  # type: ignore
        end = time()
        rounding_time = end - start

        trial = RoundingTrial(
            result.is_success(), rounding_time, result.get_optimal_cost(), result
        )
        trials.append(trial)

    plot_rounding_trials(trials, output_dir)

    best_trial_idx = np.argmin([trial.cost for trial in trials])
    best_trial = trials[best_trial_idx]

    for idx, trial in enumerate(trials):
        logger.info(
            f"Trial {idx}: {trial}, optimality gap (upper bound): {compute_optimality_gap_pct(trial.cost, relaxed_cost):.3f} %"
        )
        trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
            *trajopt.evaluate_state_input_forces(trial.result),
            sys,
            cfg.trajopt_params.T_s,
        )

        trial_dir = output_dir / f"trial_{idx}"

        if idx == best_trial_idx:
            trial_dir = Path(str(trial_dir) + "_BEST")
        trial_dir.mkdir(exist_ok=True)
        trajectory.plot(trial_dir / "trajectory.pdf")
        trajectory.animate(trial_dir / "animation.mp4")

    logger.info(f"Best trial: {best_trial_idx}")
    logger.info(
        f"Best optimality gap: {compute_optimality_gap_pct(best_trial.cost, relaxed_cost):.4f}%"
    )


def main(output_dir: Path, debug: bool, logger: Logger) -> None:
    # test_cart_pole_w_walls()
    # test_lcs_trajectory_optimization()
    # test_lcs_get_state_input_forces_from_vals()
    # test_lcs_trajopt_with_sparsity_construction()

    cart_pole_experiment_1(output_dir, debug, logger)
    # cart_pole_test_mechanical_elimination(output_dir, debug, logger)


if __name__ == "__main__":
    debug, output_dir, logger = default_script_setup()
    main(output_dir, debug, logger)