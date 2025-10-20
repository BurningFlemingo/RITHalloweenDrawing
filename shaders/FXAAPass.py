from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *
from Sampling import *

class FXAAFragmentShader:
    def __init__(self, color_attachment: Sampler2D):
        self.color_attachment = color_attachment
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        uv: Vec2 = attributes.tex_uv
        frag_color: Vec3 = self.color_attachment.sample(*uv).xyz
        luminance: float = dot(frag_color, Vec3(0.2126, 0.7152, 0.0722))
        
        return [Vec4(*luminance, 1.0)]
