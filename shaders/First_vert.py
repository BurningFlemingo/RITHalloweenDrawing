from typing import NamedTuple
from typing import Callable

from VectorMath import *
from MatrixMath import *
from RenderTypes import *


VertShaderObject = Callable[[Any], Vertex]

class FirstVertexObject:
    class Uniforms(NamedTuple):
        model_matrix: Mat4
        normal_matrix: Mat4
        view_matrix: Mat4
        projection_matrix: Mat4
        
    class Attributes(NamedTuple):
        pos: Vec3
        normal: Vec3
        tex_uv: Vec2

    class FragmentAttributes(NamedTuple):
        pos: Vec3
        normal: Vec3
        tex_uv: Vec2
    
    def __init__(self, uniforms: Uniforms):
        self.uniforms = uniforms

    def __call__(self, in_attributes: Attributes) -> Vertex:
        model_matrix, normal_matrix, view_matrix, projection_matrix = self.uniforms
        pos, normal, tex_uv = in_attributes

        view_pos: Vec4 = view_matrix * \
            model_matrix * Vec4(*pos, 1.0)

        normal: Vec3 = Vec3(*(normal_matrix * Vec4(*normal, 1.0))[:3])

        out_position = projection_matrix * view_pos
        out_attributes = self.FragmentAttributes(
            pos=Vec3(*view_pos[:3]), normal=normal, tex_uv=tex_uv)

        return Vertex(pos=out_position, fragment_attributes=out_attributes)
