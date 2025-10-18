from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetLoader import *
from Sampling import *


class SkyboxVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3

    class OutAttributes(NamedTuple):
        dir: Vec3

    def __init__(self, view_matrix: Mat4, projection_matrix: Mat4):
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix

    def __call__(self, in_attributes: Attributes) -> Vertex:
        view_matrix: Mat4 = self.view_matrix

        view_matrix: Mat4 = Mat4(
            row1=Vec4(*view_matrix.row1[:3], 0),
            row2=Vec4(*view_matrix.row2[:3], 0),
            row3=Vec4(*view_matrix.row3[:3], 0),
            row4=Vec4(0, 0, 0, 1),
        )
        projection_matrix: Mat4 = self.projection_matrix

        out_position = projection_matrix * \
            view_matrix * Vec4(*in_attributes.pos, 1.0)
        out_position = Vec4(out_position.x, out_position.y,
                            out_position.w, out_position.w)

        out_attributes = self.OutAttributes(
            dir=in_attributes.pos
        )

        return Vertex(pos=out_position, fragment_attributes=out_attributes)


class SkyboxFragmentShader:
    def __init__(self, skybox: Sampler3D):
        self.skybox = skybox

    def __call__(self, attributes: SkyboxVertexShader.OutAttributes) -> list[Vec4]:
        return [self.skybox.sample(attributes.dir) * 10]
