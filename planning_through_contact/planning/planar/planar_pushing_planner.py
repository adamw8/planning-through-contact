from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pydot
import pydrake.geometry.optimization as opt
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MathematicalProgramResult,
    MosekSolver,
    SolverOptions,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    def __init__(
        self,
        config: PlanarPlanConfig,
        contact_locations: Optional[List[PolytopeContactLocation]] = None,
    ):
        self.slider = config.dynamics_config.slider
        self.config = config

        self.source = None
        self.target = None
        self.relaxed_gcs_result = None

        if (
            self.config.non_collision_cost.avoid_object
            and config.num_knot_points_non_collision <= 2
        ):
            raise ValueError(
                "It is not possible to avoid object with only 2 knot points."
            )

        # TODO(bernhardpg): should just extract faces, rather than relying on the
        # object to only pass faces as contact locations
        self.contact_locations = contact_locations
        if self.contact_locations is None:
            self.contact_locations = self.slider.geometry.contact_locations

    def formulate_problem(self) -> None:
        assert self.config.start_and_goal is not None
        self.slider_pose_initial = self.config.start_and_goal.slider_initial_pose
        self.slider_pose_target = self.config.start_and_goal.slider_target_pose
        self.pusher_pose_initial = self.config.start_and_goal.pusher_initial_pose
        self.pusher_pose_target = self.config.start_and_goal.pusher_target_pose

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()

        # costs for non-collisions are added by each of the separate subgraphs
        for m, v in zip(self.contact_modes, self.contact_vertices):
            m.add_cost_to_vertex(v)

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        assert self.contact_locations is not None

        if not all([loc.pos == ContactLocation.FACE for loc in self.contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_plan_spec(
                loc,
                self.config,
            )
            for loc in self.contact_locations
        ]

        for mode in self.contact_modes:
            mode.add_so2_cut(
                self.slider_pose_initial.theta, self.slider_pose_target.theta
            )

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        self.edges = {}
        if self.config.allow_teleportation:
            for i, j in combinations(range(self.num_contact_modes), 2):
                self.edges[
                    (self.contact_modes[i].name, self.contact_modes[j].name)
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                    VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                    only_continuity_on_slider=True,
                )
                self.edges[
                    (self.contact_modes[j].name, self.contact_modes[i].name)
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                    VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                    only_continuity_on_slider=True,
                )
        else:
            # connect contact modes through NonCollisionSubGraphs
            connections = list(combinations(range(self.num_contact_modes), 2))

            self.subgraphs = [
                self._build_subgraph_between_contact_modes(
                    mode_i, mode_j, self.config.no_cycles
                )
                for mode_i, mode_j in connections
            ]

            if self.config.use_entry_and_exit_subgraphs:
                self.source_subgraph = self._create_entry_or_exit_subgraph("entry")
                self.target_subgraph = self._create_entry_or_exit_subgraph("exit")

        self._set_initial_poses(self.pusher_pose_initial, self.slider_pose_initial)
        self._set_target_poses(self.pusher_pose_target, self.slider_pose_target)

    def _build_subgraph_between_contact_modes(
        self,
        first_contact_mode_idx: int,
        second_contact_mode_idx: int,
        no_cycles: bool = False,
    ) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs,
            self.config,
            f"FACE_{first_contact_mode_idx}_to_FACE_{second_contact_mode_idx}",
        )
        if no_cycles:  # only connect lower idx faces to higher idx faces
            if first_contact_mode_idx <= second_contact_mode_idx:
                incoming_idx = first_contact_mode_idx
                outgoing_idx = second_contact_mode_idx
            else:  # second_contact_mode_idx <= first_contact_mode_idx
                outgoing_idx = first_contact_mode_idx
                incoming_idx = second_contact_mode_idx

            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    incoming_idx
                ),
                VertexModePair(
                    self.contact_vertices[incoming_idx],
                    self.contact_modes[incoming_idx],
                ),
                incoming=True,
                outgoing=False,
            )
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    outgoing_idx
                ),
                VertexModePair(
                    self.contact_vertices[outgoing_idx],
                    self.contact_modes[outgoing_idx],
                ),
                incoming=False,
                outgoing=True,
            )
        else:
            for idx in (first_contact_mode_idx, second_contact_mode_idx):
                subgraph.connect_with_continuity_constraints(
                    self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
                    VertexModePair(
                        self.contact_vertices[idx],
                        self.contact_modes[idx],
                    ),
                )
        return subgraph

    def _get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        all_pairs = {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.contact_vertices, self.contact_modes)
        }
        # Add all vertices from subgraphs
        if not self.config.allow_teleportation:
            for subgraph in self.subgraphs:
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        # Add source and target vertices (and possibly the ones associated
        # with the entry and exit subgraphs)
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            assert self.source is not None
            assert self.target is not None

            all_pairs[self.source.mode.name] = self.source
            all_pairs[self.target.mode.name] = self.target
        else:
            for subgraph in (self.source_subgraph, self.target_subgraph):
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        return all_pairs

    def _create_entry_or_exit_subgraph(
        self, entry_or_exit: Literal["entry", "exit"]
    ) -> NonCollisionSubGraph:
        if entry_or_exit == "entry":
            name = "ENTRY"
            kwargs = {"outgoing": True, "incoming": False}
        else:
            name = "EXIT"
            kwargs = {"outgoing": False, "incoming": True}

        subgraph = NonCollisionSubGraph.create_with_gcs(self.gcs, self.config, name)

        for idx, (vertex, mode) in enumerate(
            zip(self.contact_vertices, self.contact_modes)
        ):
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
                VertexModePair(vertex, mode),
                **kwargs,
            )
        return subgraph

    def _set_initial_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.source = self._add_single_source_or_target(
                pusher_pose, slider_pose, "initial"
            )
        else:
            self.source_subgraph.set_initial_poses(pusher_pose, slider_pose)
            self.source = self.source_subgraph.source

    def _set_target_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.target = self._add_single_source_or_target(
                pusher_pose, slider_pose, "final"
            )
        else:
            self.target_subgraph.set_final_poses(pusher_pose, slider_pose)
            self.target = self.target_subgraph.target

    def _add_single_source_or_target(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> VertexModePair:
        set_slider_pose = True
        terminal_cost = False

        mode = NonCollisionMode.create_source_or_target_mode(
            self.config,
            slider_pose,
            pusher_pose,
            initial_or_final,
            set_slider_pose=set_slider_pose,
            terminal_cost=terminal_cost,
        )
        vertex = self.gcs.AddVertex(mode.get_convex_set(), mode.name)
        pair = VertexModePair(vertex, mode)

        if terminal_cost:  # add cost on target vertex
            mode.add_cost_to_vertex(vertex)

        # connect source or target to all contact modes
        if initial_or_final == "initial":
            # source to contact modes
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[
                    ("source", contact_mode.name)
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    pair,
                    VertexModePair(contact_vertex, contact_mode),
                    only_continuity_on_slider=True,
                )
        else:  # contact modes to target
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[
                    (contact_mode.name, "target")
                ] = gcs_add_edge_with_continuity(
                    self.gcs,
                    VertexModePair(contact_vertex, contact_mode),
                    pair,
                    only_continuity_on_slider=True,
                )

        return pair

    def _get_mosek_params(
        self,
        solver_params: PlanarSolverParams,
        tolerance: float = 1e-5,
        presolve: bool = True,
    ) -> SolverOptions:
        solver_options = SolverOptions()
        if solver_params.print_solver_output:
            # solver_options.SetOption(CommonSolverOption.kPrintFileName, "optimization_log.txt")  # type: ignore
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        if solver_params.save_solver_output:
            solver_options.SetOption(CommonSolverOption.kPrintFileName, "solver_log.txt")  # type: ignore

        mosek = MosekSolver()
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tolerance
        )
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", tolerance
        )
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tolerance
        )

        solver_options.SetOption(
            mosek.solver_id(),
            "MSK_DPAR_OPTIMIZER_MAX_TIME",
            solver_params.max_mosek_solve_time,
        )

        if not presolve:
            solver_options.SetOption(mosek.solver_id(), "MSK_IPAR_PRESOLVE_USE", 0)

        return solver_options

    def _solve(self, solver_params: PlanarSolverParams) -> MathematicalProgramResult:
        """
        Returns the relaxed GCS result, potentially with non-binary flow values.
        """
        options = opt.GraphOfConvexSetsOptions()

        options.convex_relaxation = solver_params.gcs_convex_relaxation
        if solver_params.gcs_convex_relaxation:
            options.preprocessing = True
            # We want to solve only the convex relaxation first
            options.max_rounded_paths = 0

        if solver_params.solver == "mosek":
            mosek = MosekSolver()
            options.solver = mosek
            options.solver_options = self._get_mosek_params(
                solver_params, 1e-4, presolve=False
            )
        else:  # clarabel
            clarabel = ClarabelSolver()
            options.solver = clarabel
            options.solver_options.SetOption(clarabel.solver_id(), "tol_feas", 1e-4)
            options.solver_options.SetOption(clarabel.solver_id(), "tol_gap_rel", 1e-4)
            options.solver_options.SetOption(clarabel.solver_id(), "tol_gap_abs", 1e-4)

        assert self.source is not None
        assert self.target is not None

        # TODO: The following commented out code allows you to pick which path to choose
        # active_vertices = ["source", "FACE_2", "FACE_0", "target"]
        # active_edges = [
        #     self.edges[(active_vertices[i], active_vertices[i + 1])]
        #     for i in range(len(active_vertices) - 1)
        # ]
        # result = self.gcs.SolveConvexRestriction(active_edges, options)

        result = self.gcs.SolveShortestPath(
            self.source.vertex, self.target.vertex, options
        )

        if solver_params.print_flows:
            self._print_edge_flows(result)

        return result

    def get_solution_paths(
        self,
        result: MathematicalProgramResult,
        solver_params: PlanarSolverParams,
    ) -> Optional[List[PlanarPushingPath]]:
        """
        Returns N solution paths, sorted in increasing order based on optimal cost,
        where N = solver_params.rounding_steps.
        """
        assert self.source is not None
        assert self.target is not None

        options = opt.GraphOfConvexSetsOptions()
        options.max_rounded_paths = solver_params.rounding_steps
        options.max_rounding_trials = solver_params.max_rounding_trials

        options.convex_relaxation = True
        options.preprocessing = True

        options.solver_options = self._get_mosek_params(solver_params, 1e-5)

        paths, results = self.gcs.GetRandomizedSolutionPath(
            self.source.vertex, self.target.vertex, result, options
        )

        flows = [result.GetSolution(e.phi()) for e in self.gcs.Edges()]

        # Sort the paths and results by optimal cost
        paths_and_results = zip(paths, results)
        only_successful_res = [
            pair for pair in paths_and_results if pair[1].is_success()
        ]

        if len(only_successful_res) == 0:
            print("No GCS trajectories rounded succesfully")
            return None

        # if len(only_successful_res) == 0:
        #     raise RuntimeError("No trajectories rounded succesfully")

        sorted_res = sorted(
            only_successful_res, key=lambda pair: pair[1].get_optimal_cost()
        )
        paths, results = zip(*sorted_res)

        paths = [
            PlanarPushingPath.from_path(
                self.gcs,
                result,
                path,
                self._get_all_vertex_mode_pairs(),
                assert_nan_values=solver_params.assert_nan_values,
            )
            for path, result in zip(paths, results)
        ]
        # print(f"Found {len(paths)} paths after GCS rounding.")

        return paths

    def _plan_paths(
        self, solver_params: PlanarSolverParams
    ) -> Optional[List[PlanarPushingPath]]:
        """
        Plans a path.
        """
        assert self.source is not None
        assert self.target is not None

        gcs_result = self._solve(solver_params)
        self.relaxed_gcs_result = gcs_result

        if solver_params.assert_result:
            assert gcs_result.is_success()
        else:
            if not gcs_result.is_success():
                print("WARNING: Solver did not find a solution!")

        if not gcs_result.is_success():
            return None

        if solver_params.measure_solve_time:
            print(
                f"Total elapsed optimization time: {gcs_result.get_solver_details().optimizer_time}"
            )

        if solver_params.print_cost:
            cost = gcs_result.get_optimal_cost()
            print(f"Cost: {cost}")

        # Get N paths from GCS rounding, pick the best one
        paths = self.get_solution_paths(
            gcs_result,
            solver_params,
        )

        if paths is None:
            print("No gcs paths found")
            return None
        else:
            if solver_params.print_rounding_details:
                print(f"num rounded paths: {len(paths)}")

            return paths

    def _get_rounded_paths(
        self, solver_params: PlanarSolverParams, paths: List[PlanarPushingPath]
    ) -> Optional[List[PlanarPushingPath]]:
        if solver_params.rounding_steps > 0:
            for path in paths:
                path.do_rounding(solver_params)

            feasible_paths = [
                p
                for p in paths
                if p.rounded_result is not None and p.rounded_result.is_success()
            ]

            if solver_params.print_rounding_details:
                print(f"num rounded feasible paths: {len(feasible_paths)}")

            if len(feasible_paths) == 0:
                return None
            else:
                return feasible_paths
        else:
            raise NotImplementedError("Must enable rounding steps")

    def _pick_best_path(self, paths: List[PlanarPushingPath]) -> PlanarPushingPath:
        rounded_costs = [
            p.rounded_result.get_optimal_cost()
            for p in paths
            if p.rounded_result is not None  # type
        ]

        best_idx = np.argmin(rounded_costs)
        path = paths[best_idx]

        return path

    def pick_top_k_paths(
        self, paths: List[PlanarPushingPath], k: int
    ) -> PlanarPushingPath:
        rounded_costs = [
            p.rounded_result.get_optimal_cost()
            for p in paths
            if p.rounded_result is not None  # type
        ]
        # if k is -1, return all paths (sorted), otherwise return min(k, len(paths)) paths
        k = min(k, len(rounded_costs)) if k > 0 else len(rounded_costs)
        best_idxs = np.argsort(rounded_costs)[:k]
        best_paths = [paths[idx] for idx in best_idxs]

        return best_paths

    def plan_path(
        self, solver_params: PlanarSolverParams
    ) -> Optional[PlanarPushingPath]:
        paths = self._plan_paths(solver_params)
        if paths is None:
            return None

        feasible_paths = self._get_rounded_paths(solver_params, paths)
        if feasible_paths is None:
            print("No feasible path found after nonlinear rounding!")
            return None
        print(f"num feasible paths: {len(feasible_paths)}")

        self.path = self._pick_best_path(feasible_paths)

        if solver_params.print_path:
            print(f"path: {self.path.get_path_names()}")

        return self.path

    def plan_multiple_paths(
        self, solver_params: PlanarSolverParams
    ) -> List[PlanarPushingPath]:
        paths = self._plan_paths(solver_params)
        if paths is None:
            return None

        feasible_paths = self._get_rounded_paths(solver_params, paths)
        if feasible_paths is None:
            print("Failed to find feasible paths after rounding!")
            return None

        return feasible_paths

    def _print_edge_flows(self, result: MathematicalProgramResult) -> None:
        """
        Used for debugging.
        """
        edge_phis = {
            (e.u().name(), e.v().name()): result.GetSolution(e.phi())
            for e in self.gcs.Edges()
        }
        sorted_flows = sorted(edge_phis.items(), key=lambda item: item[0])
        for name, flow in sorted_flows:
            print(f"{name}: {flow}")

    def create_graph_diagram(
        self,
        filename: Optional[str] = None,
        result: Optional[MathematicalProgramResult] = None,
    ) -> pydot.Dot:
        """
        Optionally saves the graph to file if a string is given for the 'filepath' argument.
        """
        graphviz = self.gcs.GetGraphvizString(
            precision=2, result=result, show_slacks=False
        )

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        if filename is not None:
            data.write_png(filename + ".png")

        return data
