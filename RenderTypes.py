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
    def __init__(self, pos: Vec3,color: Vec3, intensity: float=1.0):
        self.pos = Vec3(*pos)

        self.intensity = intensity
        
        self.color = Vec3(*color)
        
class DirectionalLight:
    def __init__(self, dir: Vec3, color: Vec3, intensity: float=1.0):
        self.dir = Vec3(*dir)

        self.intensity = intensity
        
        self.color = Vec3(*color)

class SpotLight:
    def __init__(self, pos: Vec3, dir: Vec3, inner_cutoff_angle: float, outer_cutoff_angle: float, color: Vec3, intensity: float=1.0):
        """
            cutoff_angle is in degrees
        """
        self.pos = Vec3(*pos)
        self.dir = Vec3(*dir)
        
        self.inner_cutoff_angle = inner_cutoff_angle
        self.outer_cutoff_angle = outer_cutoff_angle
        
        self.cos_inner_cutoff = math.cos(math.radians(inner_cutoff_angle)) 
        self.cos_outer_cutoff = math.cos(math.radians(outer_cutoff_angle)) 

        self.intensity = intensity
        
        self.color = Vec3(*color)
