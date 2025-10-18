from typing import NamedTuple
from enum import IntEnum
from VectorMath import *
from Buffer import *
from AssetLoader import *


class Cubemap(NamedTuple):
    class Face(IntEnum):
        RIGHT = 0
        LEFT = 1
        TOP = 2
        BOTTOM = 3
        FRONT = 4
        BACK = 5

    faces: list[Buffer]

def dir_to_uv(dir: Vec3) -> tuple[Vec2, Cubemap.Face]:
    dir = normalize(dir)

    abs_x: float = abs(dir.x)
    abs_y: float = abs(dir.y)
    abs_z: float = abs(dir.z)

    uv: Vec2 | None = None
    face: Cubemap.Face | None = None
    major_axis: float = 1

    if (abs_x >= abs_y and abs_x >= abs_z):
        major_axis = abs_x
        if (dir.x >= 0):
            face = Cubemap.Face.RIGHT
            uv = Vec2(-dir.z, dir.y)
        else:
            face = Cubemap.Face.LEFT
            uv = Vec2(dir.z, dir.y)
    if (abs_y >= abs_x and abs_y >= abs_z):
        major_axis = abs_y
        if (dir.y >= 0):
            face = Cubemap.Face.TOP
            uv = Vec2(dir.x, -dir.z)
        else:
            face = Cubemap.Face.BOTTOM
            uv = Vec2(dir.x, dir.z)
    if (abs_z >= abs_x and abs_z >= abs_y):
        major_axis = abs_z
        if (dir.z >= 0):
            face = Cubemap.Face.FRONT
            uv = Vec2(dir.x, dir.y)
        else:
            face = Cubemap.Face.BACK
            uv = Vec2(-dir.x, dir.y)

    assert uv != None
    assert face != None
    
    uv = ((uv / major_axis) + 1) / 2
    return (uv, face)
