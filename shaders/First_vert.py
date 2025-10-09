from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *

from shaders.First_frag import FragmentAttributes

class VertexAttributes(NamedTuple):
    pos: Vec3 
    normal: Vec3
    tex_uv: Vec2

class VertexUniforms(NamedTuple):
    model_matrix: Mat4
    normal_matrix: Mat4
    view_matrix: Mat4
    projection_matrix: Mat4
    
def vertex_shader(uniforms: VertexUniforms, attributes: VertexAttributes) -> Vertex:
    pos, normal, tex_uv = attributes
    model_matrix, normal_matrix, view_matrix, projection_matrix = uniforms
    
    view_pos: Vec4 = view_matrix * \
        model_matrix * Vec4(*pos, 1.0)

    normal: Vec3 = Vec3(*(normal_matrix * Vec4(*normal, 1.0))[:3])

    out_position = projection_matrix * view_pos
    out_attributes = FragmentAttributes(pos=Vec3(*view_pos[:3]), normal=normal, tex_uv=tex_uv)
    
    return Vertex(pos=out_position, fragment_attributes=out_attributes)
