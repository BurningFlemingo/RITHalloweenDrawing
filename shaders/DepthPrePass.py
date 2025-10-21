from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from shaders.Lighting import *
from Rasterizer import *


class DepthPrePassVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3

    def __init__(self, model_matrix: Mat4, view_matrix: Mat4, projection_matrix: Mat4):
        self.model_matrix = model_matrix
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix

    def __call__(self, in_attributes: Attributes) -> Vertex:
        pos: Vec3 = in_attributes.pos
        model_matrix: Mat4 = self.model_matrix
        view_matrix: Mat4 = self.view_matrix
        projection_matrix: Mat4 = self.projection_matrix
        
        out_position = projection_matrix * view_matrix * model_matrix * Vec4(*pos, 1.0)
        return Vertex(pos=out_position, fragment_attributes=None)
