from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *
from Sampling import *

class GaussianFragmentShader:
    def __init__(self, color_attachment: Sampler2D, horizontal: bool):
        self.color_attachment = color_attachment
        self.horizontal = horizontal
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        weights: list[float] = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216]

        image: Sampler2D = self.color_attachment
        uv: Vec2 = attributes.uv

        fragment_color: Vec4 = image.sample(*uv, WrappingMode.CLAMP) * weights[0]
        texel_step: Vec2 = Vec2(1/image.get_size().width, 1/image.get_size().height)
        if (self.horizontal):
            v: float = uv.y
            for i in range(1, 5):
                u0: float = uv.x + (i * texel_step.x)
                u1: float = uv.x - (i * texel_step.x)
                fragment_color += image.sample(u0, v, WrappingMode.CLAMP) * weights[0]
                fragment_color += image.sample(u1, v, WrappingMode.CLAMP) * weights[0]
        else:
            u: float = uv.x
            for i in range(1, 5):
                v0: float = uv.y + (i * texel_step.y)
                v1: float = uv.y - (i * texel_step.y)
                fragment_color += image.sample(u, v0, WrappingMode.CLAMP) * weights[0]
                fragment_color += image.sample(u, v1, WrappingMode.CLAMP) * weights[0]
        

        return [fragment_color]
