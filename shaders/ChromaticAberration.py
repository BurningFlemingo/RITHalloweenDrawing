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
        
        dir: Vec2 = (uv - Vec2(0.5, 0.5))
        if (dir.y < 0):
            dir *= Vec2(1.0, -1.0)
            
        # distance: float = dir.magnitude()
        
        red_offset: Vec2 = dir * 0.025
        green_offset: Vec2 = dir * 0.01
        blue_offset: Vec2 = dir * -0.0150
        
        red: float = screen_tex.sample(*(uv + red_offset), mode=WrappingMode.CLAMP).x
        green: float = screen_tex.sample(*(uv + green_offset), mode=WrappingMode.CLAMP).y
        blue: float = screen_tex.sample(*(uv + blue_offset), mode=WrappingMode.CLAMP).z
        
        frag_color: Vec4 = Vec4(red, green, blue, 1.0)
        
        return [frag_color]
