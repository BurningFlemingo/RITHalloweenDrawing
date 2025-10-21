from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *
from Sampling import *

class TonemapFragmentShader:
    def __init__(self, color_attachment: Sampler2D):
        self.color_attachment = color_attachment
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        a: float = 0.15
        b: float = 0.50
        c: float = 0.10
        d: float = 0.20
        e: float = 0.02
        f: float = 0.30
        w: float = 11.2
        
        # bloom_color: Vec3 = Vec3(*self.bloom_attachment.sampleUV(*attributes.uv)[:3])
        hdr_color: Vec3 = self.color_attachment.sample(*attributes.tex_uv).xyz
        mapped: Vec3 = ((hdr_color * (hdr_color * a + c * b) + d * e) / (hdr_color * (hdr_color * a + b) + d * f)) -e/f
        
        gamma: float = 2.2
        # rgb luma coefficients from the Rec. 709 Standard
        luminance: float = dot(mapped ** 1/gamma, Vec3(0.2126, 0.7152, 0.0722))
        
        return [Vec4(*mapped, luminance)]
