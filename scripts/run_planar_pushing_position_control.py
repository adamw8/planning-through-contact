from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_position_control import (
    PlanarPushingPositionControl,
)


def run_position_control(debug: bool = False):
    traj = PlanarPushingTrajectory.load("box_pushing.pkl")

    position_control_node = PlanarPushingPositionControl(traj)
    if debug:
        position_control_node.export_diagram("position_control_diagram.png")

    position_control_node.run()


if __name__ == "__main__":
    run_position_control(debug=True)