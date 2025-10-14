from typing import NamedTuple
from enum import Enum
from VectorMath import *
from Buffer import *
from AssetLoader import *


class Cubemap(NamedTuple):
    class Face(Enum):
        RIGHT = 0,
        LEFT = 1,
        TOP = 2,
        BOTTOM = 3,
        FRONT = 4,
        BACK = 5,
    
    faces: list[Buffer]

    def sample(self, dir: Vec3) -> Vec4:
        dir = normalize(dir)

        abs_x: float = abs(dir.x)
        abs_y: float = abs(dir.y)
        abs_z: float = abs(dir.z)

        uv: Vec2 | None = None 
        face: Face | None = None
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

        uv = ((uv / major_axis) + 1) / 2

        return self.faces[face].sample(*uv)


def load_cubemap(dir_path: str) -> Cubemap:
    face_strings: list[str] = [
        "right", "left", "top", "bottom", "front", "back"        
    ]
    faces: list[Buffer] = []
    for face_string in face_strings:
        path: str = dir_path + face_string + ".bmp"
        face: Buffer = load_bmp(path, ColorSpace.SRGB, ColorSpace.LINEAR)
        faces.append(face)
    
    return Cubemap(faces=faces)
