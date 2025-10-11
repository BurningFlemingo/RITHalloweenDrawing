from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetLoader import *


class ShadowPassVertexShader:
    class Uniforms(NamedTuple):
        model_matrix: Mat4
        light_space_matrix: Mat4

    class Attributes(NamedTuple):
        pos: Vec3

    def __init__(self, uniforms: Uniforms):
        self.uniforms = uniforms

    def __call__(self, attributes: Attributes) -> Vertex:
        uniforms: Uniforms = self.uniforms
        
        model_matrix: Mat4 = uniforms.model_matrix
        light_space_matrix: Mat4 = uniforms.light_space_matrix
        
        pos = attributes.pos

        out_position: Vec4 = light_space_matrix * model_matrix * Vec4(*pos, 1.0)
        return Vertex(pos=out_position, fragment_attributes=None)
