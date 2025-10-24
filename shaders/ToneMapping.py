# MIT License
# 
# Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
#   https://github.com/sobotka/AgX
#
# https://iolite-engine.com/blog_posts/minimal_agx_implementation

from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *
from Sampling import *

def agx_sigmoid_contrast_approx(x: Vec3):
    x2: Vec3 = x * x
    x4: Vec3 = x2 * x2
    x6: Vec3 = x4 * x2
  
    return - 17.86   * x6 * x \
         + 78.01     * x6     \
         - 126.7     * x4 * x \
         + 92.06     * x4     \
         - 28.72     * x2 * x \
         + 4.361     * x2     \
         - 0.1718    * x      \
         + 0.002857

def color_correction(val: Vec3) -> Vec3:
    # ASC CDL 
    
    # normal
    offset: Vec3 = Vec3(0.0, 0.0, 0.0);
    slope: Vec3 = Vec3(1.0, 1.0, 1.0);
    power: Vec3 = Vec3(1.0, 1.0, 1.0);
    sat: float = 1.0;

    # golden
    # slope = Vec3(1.0, 0.9, 0.5);
    # power = Vec3(0.8, 0.8, 0.8);
    # sat = 0.8;

    # punchy
    # power: Vec3 = Vec3(1.35, 1.35, 1.35);
    # sat: float = 1.4
    
    val = val * slope + offset
    val = Vec3(
        val.x ** power.x,
        val.y ** power.y,
        val.z ** power.z
    )

    # rgb luma coefficients from the Rec. 709 Standard
    rec_709_primaries: Vec3 = Vec3(0.2126, 0.7152, 0.0722)
    luminance: float = dot(val, rec_709_primaries)

    return (sat * (val - luminance)) + luminance

    
def agx(val: Vec3) -> Vec3:
    agx_inset: Mat4 = Mat4(
        Vec4(0.842479062253094, 0.0423282422610123, 0.0423756549057051, 0.0),
        Vec4(0.0784335999999992,  0.878468636469772,  0.0784336, 0.0),
        Vec4(0.0792237451477643, 0.0791661274605434, 0.879142973793104, 0.0), 
        Vec4(0.0, 0.0, 0.0, 1.0)
    )

    agx_outset: Mat4 = Mat4(
        Vec4(1.19687900512017, -0.0528968517574562, -0.0529716355144438, 0.0),
        Vec4(-0.0980208811401368, 1.15190312990417, -0.0980434501171241, 0.0),
        Vec4(-0.0990297440797205, -0.0989611768448433, 1.15107367264116, 0.0), 
        Vec4(0.0, 0.0, 0.0, 1.0)
    )


    # from https://iolite-engine.com/blog_posts/minimal_agx_implementation
    # min_ev: float = -12.47393
    # max_ev: float = 4.026069
    white_point: float = 16.2917
    black_point: float = 0.00017578
    
    min_ev: float = math.log2(black_point)
    max_ev: float = math.log2(white_point)
    
    two_exp_min_ev: float = 2 ** min_ev 
    
    val = (agx_inset * Vec4(*val, 1.0)).xyz
    val = Vec3(
        max(val.x, two_exp_min_ev),
        max(val.y, two_exp_min_ev),
        max(val.z, two_exp_min_ev)
    )
    
    val = Vec3(
        min(math.log2(val.x), max_ev),
        min(math.log2(val.y), max_ev),
        min(math.log2(val.z), max_ev)
    )
    val = (val - min_ev) / (max_ev - min_ev)
    val = agx_sigmoid_contrast_approx(val)
    val = color_correction(val)

    val = (agx_outset * Vec4(*val, 1.0)).xyz
    val = Vec3(
        min(max(val.x, 0), 1), 
        min(max(val.y, 0), 1), 
        min(max(val.z, 0), 1), 
    )
    return val


class TonemapFragmentShader:
    def __init__(self, color_attachment: Sampler2D, neutral_scene_luminance: float, key: float):
        print("neutral_luminance:", neutral_scene_luminance)
        self.color_attachment = color_attachment
        self.exposure = key / (neutral_scene_luminance * 2)
        print("exposure:", self.exposure, "key:", key)
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        uv: Vec2 = attributes.tex_uv
        linear_color: Vec3 = self.color_attachment.sample(*uv).xyz
        linear_color *= self.exposure 

        mapped_color: Vec3 = agx(linear_color)

        return [Vec4(*mapped_color, 1.0)]
