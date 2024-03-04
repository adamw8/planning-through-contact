import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.all import RigidTransform, AbstractValue, MultibodyPlant
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing.inverse_kinematics import solve_ik

class InverseKinematicsSystem(LeafSystem):
    """Solves inverse kinematics"""

    def __init__(self, plant: MultibodyPlant,
                 default_joint_positions: np.ndarray,
                 disregard_angle: bool = False):
        super().__init__()
        if default_joint_positions:
            assert len(default_joint_positions) == plant.num_positions()
        
        self._plant = plant
        self._default_joint_positions = default_joint_positions
        self._disregard_angle = disregard_angle

        self.DeclareAbstractInputPort(
            "rigid_transform_input", AbstractValue.Make(RigidTransform())
        )
        self.DeclareVectorOutputPort("q", plant.num_positions(), self.DoCalcVectorOutput)

    def DoCalcVectorOutput(self, context, output):
        rigid_transform = self.EvalAbstractInput(context, 0).get_value()
        # Hack to fix initialization error
        # TODO: fix this properly (set initial output port value in input systems)
        if np.allclose(rigid_transform.translation()[:2], np.zeros(2)):
            rigid_transform.set_translation(np.array([0.5, 0.25, 0]))
        q = solve_ik(self._plant,
                     rigid_transform,
                     self._default_joint_positions,
                     self._disregard_angle
        )
        output.SetFromVector(q)
