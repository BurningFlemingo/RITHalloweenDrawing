from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *

class QuadVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3

    class OutAttributes(NamedTuple):
        tex_uv: Vec2

    def __call__(self, in_attributes: Attributes) -> Vertex:
        out_position = Vec4(*in_attributes.pos, 1.0)
        u: float = (in_attributes.pos.x + 1) / 2
        v: float = (in_attributes.pos.y + 1) / 2
        out_attributes = self.OutAttributes(tex_uv=Vec2(u, v))

        return Vertex(pos=out_position, fragment_attributes=out_attributes)

