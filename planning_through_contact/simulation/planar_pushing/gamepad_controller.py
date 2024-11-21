import numpy as np
from pydrake.all import StartMeshcat

# Pydrake imports
from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


class GamepadController(LeafSystem):
    def __init__(
        self,
        meshcat,
        translation_scale: float,
        deadzone: float,
        gamepad_orientation: np.ndarray,
    ):
        super().__init__()

        self.translation_scale = translation_scale
        self.deadzone = deadzone
        self.gamepad_orientation = gamepad_orientation

        self.init_xy = None
        self.prev_button_values = [0.0 for _ in range(17)]
        self.button_index = {
            "A": 0,
            "B": 1,
            "X": 2,
            "Y": 3,
            "LB": 4,
            "RB": 5,
            "LT": 6,
            "RT": 7,
            "BACK": 8,
            "START": 9,
            "LS": 10,
            "RS": 11,
            "UP": 12,
            "DOWN": 13,
            "LEFT": 14,
            "RIGHT": 15,
            "LOGO": 16,
        }

        # Set up ports
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )

        self.output = self.DeclareVectorOutputPort(
            "planar_position_command", 2, self.DoCalcOutput
        )

        # Wait for gamepad connection
        print("Gamepad meshcat (must be opened to connect gamepad)")
        self.meshcat = meshcat
        print("\nPlease connect gamepad.")
        while self.meshcat.GetGamepad().index is None:
            continue
        print("Gamepad connected.")

    def DoCalcOutput(self, context: Context, output):
        # Read in pose
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        curr_xy = PlanarPose.from_pose(pusher_pose).pos().reshape(2)

        # Get offset from gamepad
        xy_offset = self.get_xy_offset()

        # Compute target pose
        if self.init_xy is None:
            self.init_xy = curr_xy
        target_xy = self.init_xy + xy_offset
        self.init_xy = target_xy

        output.SetFromVector(target_xy)

    def get_xy_offset(self):
        gamepad = self.meshcat.GetGamepad()
        position = self.create_stick_dead_zone(gamepad.axes[0], gamepad.axes[1])
        return self.translation_scale * self.gamepad_orientation @ position

    def create_stick_dead_zone(self, x, y):
        stick = np.array([x, y])
        m = np.linalg.norm(stick)

        if m < self.deadzone:
            return np.array([0, 0])
        over = (m - self.deadzone) / (1 - self.deadzone)
        return stick * over / m
