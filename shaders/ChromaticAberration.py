from math import nan
from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from shaders.Lighting import *
from shaders.Quad import *
from Rasterizer import *


class ChromaticAberrationFragmentShader:
    def __init__(self, texture: Sampler2D):
        self.screen_tex = texture


    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        uv: Vec2 = attributes.tex_uv
        screen_tex: Sampler2D = self.screen_tex
        
        dir: Vec2 = (uv - Vec2(0.5))
        distance: float = dir.magnitude()
        
        red_offset: Vec2 = dir * distance ** 3 * 0.0125 * 8 
        green_offset: Vec2 = dir * distance ** 3 * 0.0050 * 8
        blue_offset: Vec2 = dir * distance ** 3 * -0.0075 * 8
        
        red: float = screen_tex.sample(*(uv + red_offset), mode=WrappingMode.CLAMP).x
        green: float = screen_tex.sample(*(uv + green_offset), mode=WrappingMode.CLAMP).y
        blue: float = screen_tex.sample(*(uv + blue_offset), mode=WrappingMode.CLAMP).z
        
        frag_color: Vec4 = Vec4(red, green, blue, 1.0)
        
        return [frag_color]
