from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *

class GaussianFragmentShader:
    def __init__(self, color_attachment: Buffer, horizontal: bool):
        self.color_attachment = color_attachment
        self.horizontal = horizontal
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        weights: list[float] = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216]

        image: Buffer = self.color_attachment
        uv: Vec2 = attributes.uv

        fragment_color: Vec4 = image.sampleUV(*uv, WrappingMode.CLAMP) * weights[0]
        texel_step: Vec2 = Vec2(1/image.width, 1/image.height)
        if (self.horizontal):
            v: float = uv.y
            for i in range(1, 5):
                u0: float = uv.x + (i * texel_step.x)
                u1: float = uv.x - (i * texel_step.x)
                fragment_color += image.sampleUV(u0, v, WrappingMode.CLAMP) * weights[0]
                fragment_color += image.sampleUV(u1, v, WrappingMode.CLAMP) * weights[0]
        else:
            u: float = uv.x
            for i in range(1, 5):
                v0: float = uv.y + (i * texel_step.y)
                v1: float = uv.y - (i * texel_step.y)
                fragment_color += image.sampleUV(u, v0, WrappingMode.CLAMP) * weights[0]
                fragment_color += image.sampleUV(u, v1, WrappingMode.CLAMP) * weights[0]
        

        return [fragment_color]
