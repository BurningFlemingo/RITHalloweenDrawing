from typing import NamedTuple
from Buffer import *
from VectorMath import *

import math


class Vertex(NamedTuple):
    pos: Vec4
    fragment_attributes: tuple


class Transform:
    def __init__(self, pos: Vec3, rot: Vec3 = Vec3(0.0, 0.0, 0.0), scale: Vec3 = Vec3(1.0, 1.0, 1.0)):
        """
            rot is in euler angles in degrees
        """
        self.pos = Vec3(*pos)
        self.rot = make_euler_rotor(
            Vec3(*[math.radians(theta) for theta in rot]))
        self.scale = Vec3(*scale)


class PointLight:
    def __init__(self, pos: Vec3, linear_att: float, quadratic_att: float, color: Vec3, specular: Vec3):
        self.pos = Vec3(*pos)

        self.linear_att = linear_att
        self.quadratic_att = quadratic_att
        
        self.color = Vec3(*color)
        self.specular = Vec3(*specular)
        
