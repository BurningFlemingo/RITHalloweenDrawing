from typing import NamedTuple
from Buffer import *
from VectorMath import *


class Vertex(NamedTuple):
    pos: Vec4
    fragment_attributes: tuple


class Transform(NamedTuple):
    pos: Vec3  # position
    rot: Rot3  # rotor
    scale: Vec3


class PointLight(NamedTuple):
    pos: Vec3

    linear_att: float
    quadratic_att: float

    color: Vec3  # used as the ambient and diffuse color
    specular: Vec3
