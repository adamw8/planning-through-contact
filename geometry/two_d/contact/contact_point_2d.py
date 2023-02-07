from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore

from geometry.hyperplane import Hyperplane
from geometry.two_d.box_2d import RigidBody2d
from geometry.two_d.contact.types import PolytopeContactLocation, ContactPosition
from tools.types import NpExpressionArray, NpFormulaArray, NpVariableArray


class ContactPoint2d:
    def __init__(
        self,
        body: RigidBody2d,
        contact_location: PolytopeContactLocation,
        friction_coeff: float = 0.5,
        name: str = "unnamed",
    ) -> None:
        self.name = name
        self.body = body
        self.friction_coeff = friction_coeff
        self.contact_location = contact_location

        self.normal_force = sym.Variable(f"{self.name}_c_n")
        self.friction_force = sym.Variable(f"{self.name}_c_f")
        self.normal_vec, self.tangent_vec = body.get_norm_and_tang_vecs_from_location(
            contact_location
        )

        self.contact_position = self._set_contact_position()

    def _set_contact_position(
        self,
    ) -> Union[npt.NDArray[np.float64], NpExpressionArray]:
        if self.contact_location.pos == ContactPosition.FACE:
            self.lam = sym.Variable(f"{self.name}_lam")
            vertices = self.body.get_proximate_vertices_from_location(
                self.contact_location
            )
            return self.lam * vertices[0] + (1 - self.lam) * vertices[1]
        else:
            # Get first element as we know this will only be one vertex
            corner_vertex = self.body.get_proximate_vertices_from_location(
                self.contact_location
            )[0]
            return corner_vertex

    @property
    def contact_force(self) -> NpExpressionArray:
        return (
            self.normal_force * self.normal_vec + self.friction_force * self.tangent_vec
        )

    @property
    def variables(self) -> NpVariableArray:
        if self.contact_location.pos == ContactPosition.FACE:
            return np.array([self.normal_force, self.friction_force, self.lam])
        else:
            return np.array([self.normal_force, self.friction_force])

    def create_friction_cone_constraints(self) -> NpFormulaArray:
        upper_bound = self.friction_force <= self.friction_coeff * self.normal_force
        lower_bound = -self.friction_coeff * self.normal_force <= self.friction_force
        normal_force_positive = self.normal_force >= 0
        return np.vstack([upper_bound, lower_bound, normal_force_positive])

    def get_neighbouring_vertices(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.body.get_neighbouring_vertices(self.contact_location)

    def get_contact_hyperplane(
        self,
    ) -> Hyperplane:
        return self.body.get_hyperplane_from_location(self.contact_location)