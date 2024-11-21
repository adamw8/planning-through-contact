import numpy as np
from pydrake.all import Demultiplexer, DiagramBuilder, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.planar_pushing.gamepad_controller import (
    GamepadController,
)


class GamepadControllerSource(DesiredPlanarPositionSourceBase):
    def __init__(
        self,
        meshcat,
        # translation_scale: float,
        # deadzone: float,
        # gamepad_orientation: np.ndarray,
    ):
        super().__init__()

        builder = DiagramBuilder()

        # Gamepad Controller
        self._gamepad_controller = builder.AddNamedSystem(
            "GamepadController",
            GamepadController(
                meshcat=meshcat,
                translation_scale=0.0001,
                deadzone=0.05,
                gamepad_orientation=np.array([[1, 0], [0, -1]]),
            ),
        )

        # Export inputs and outputs (external)
        builder.ExportInput(
            self._gamepad_controller.GetInputPort("pusher_pose_measured"),
            "pusher_pose_measured",
        )
        builder.ExportOutput(
            self._gamepad_controller.get_output_port(), "planar_position_command"
        )

        builder.BuildInto(self)
