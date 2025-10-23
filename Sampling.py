from typing import NamedTuple
from typing import Any
from enum import Enum
from VectorMath import *
from Buffer import *
from Cubemap import *
from dataclasses import dataclass, field

class FilterMethod(Enum):
    NEAREST = 0,
    BILINEAR = 1,
    TRILINEAR = 1,


class SizeDimensions(NamedTuple):
    width: int
    height: int

@dataclass
class Sampler2D:
    buffers: list[Buffer] = field(default_factory=list)

    min_filtering_method: FilterMethod = FilterMethod.TRILINEAR
    mag_filtering_method: FilterMethod = FilterMethod.NEAREST
    
    duvdx: Vec2 = Vec2()
    duvdy: Vec2 = Vec2()

    anisotropic_filtering: bool = False
    max_anisotropy: int = 16

    def get_size(self) -> SizeDimensions:
        return SizeDimensions(self.buffers[0].width, self.buffers[0].height)

    def calc_lod(self):
        lod: float = 0
        
        duvdx: Vec2 = self.duvdx * self.buffers[0].width
        duvdy: Vec2 = self.duvdy * self.buffers[0].height

        if (duvdx.x != 0 and duvdx.y != 0 and duvdy.x != 0 and duvdy.y != 0):
            d: float = max(duvdx.magnitude(), duvdy.magnitude())
            
            lod_bias: float = -0.5
            lod = max(min(math.log2(d) + lod_bias, len(self.buffers) - 1), 0)
            
        return lod

    def sample_anisotropic(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
        duvdx: Vec2 = self.duvdx * self.buffers[0].width
        duvdy: Vec2 = self.duvdy * self.buffers[0].height
        
        duvdx_mag: float = duvdx.magnitude()
        duvdy_mag: float = duvdy.magnitude()

        if (duvdx_mag == 0 or duvdy_mag == 0):
            return self.sample_isotropic(u, v, mode, border_color)

        normalized_duvdx: Vec2 = duvdx / duvdx_mag
        normalized_duvdy: Vec2 = duvdy / duvdy_mag

        if (normalized_duvdx == normalized_duvdy):
            return self.sample_isotropic(u, v, mode, border_color)
            
        if (dot(normalized_duvdx, normalized_duvdy) != 0):
            # https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#LODCalculation
            A: float = duvdx.y ** 2 + duvdy.y ** 2
            B: float = -2 * (duvdx.x * duvdx.y + duvdy.x * duvdy.y)
            C: float = duvdx.x ** 2 + duvdy.x ** 2
            F: float = (duvdx.x * duvdy.y - duvdy.x * duvdx.y) ** 2

            p: float = A - C
            q: float = A + C
            t: float = math.sqrt(p ** 2 + B ** 2)

            sgn_B: float = 1 if B >= 0 else -1

            duvdx = Vec2(
                math.sqrt(F * (t+p) / (t * (q+t))), 
                math.sqrt(F * (t-p) / (t * (q+t))) * sgn_B
            )

            duvdy = Vec2(
                math.sqrt(F * (t-p) / (t * (q-t))) * -sgn_B, 
                math.sqrt(F * (t+p) / (t * (q-t)))
            )

        major_axis: Vec2 = duvdy if duvdy.magnitude() >= duvdx.magnitude() else duvdx
        minor_axis: Vec2 = duvdx if duvdx != major_axis else duvdy
        
        major_axis_mag: float = major_axis.magnitude()
        minor_axis_mag: float = minor_axis.magnitude()

        ratio_of_anisotropy: float = major_axis_mag / minor_axis_mag
        n_samples: int = min(round(ratio_of_anisotropy), self.max_anisotropy)
        
        if (n_samples <= 1):
            return self.sample_isotropic(u, v, mode, border_color)
        
        if (mode == WrappingMode.NONE):
            mode = WrappingMode.CLAMP

        major_axis_uv: Vec2 = major_axis / Vec2(self.buffers[0].width, self.buffers[0].height)
        major_axis_step: Vec2 = major_axis_uv / (n_samples - 1)

        lod_bias: float = -0.5
        lod = max(min(math.log2(minor_axis_mag) + lod_bias, len(self.buffers) - 1), 0)

        uv: Vec2 = Vec2(u, v) - (major_axis_uv / 2)
        accumulated_sample: Vec4 = Vec4(0, 0, 0, 0) 
        for _ in range(0, n_samples): 
            accumulated_sample += self.sample_isotropic(*uv, mode, border_color, lod)
            uv += major_axis_step
            
        return accumulated_sample / n_samples

    def sample_isotropic(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None, lod: float | None = None):
        if (len(self.buffers) == 1):
            lod = 0
        elif (lod is None):
            lod = self.calc_lod()
            
        if (self.min_filtering_method == FilterMethod.TRILINEAR):
            lod_a: float = math.floor(lod)
            lod_b: float = math.ceil(lod)
            
            buffer_a = self.buffers[lod_a]
            buffer_b = self.buffers[lod_b]
            
            sample_a: Vec4 = sample2D(buffer_a, u, v, FilterMethod.BILINEAR, mode, border_color)
            
            if (lod_a == lod_b):
                return sample_a
            
            sample_b: Vec4 = sample2D(buffer_b, u, v, FilterMethod.BILINEAR, mode, border_color)
            
            t: float = (lod - lod_a) / (lod_b - lod_a)
            return sample_a + t * (sample_b - sample_a)
        
        buffer = self.buffers[math.floor(lod)]
        return sample2D(buffer, u, v, self.min_filtering_method, mode, border_color)

    def sample(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
        if (self.anisotropic_filtering):
            return self.sample_anisotropic(u, v, mode, border_color)
        
        return self.sample_isotropic(u, v, mode, border_color)


    def generate_mipmaps(self):
        if (len(self.buffers) > 1):
            return self
            
        num_mip_levels: int = math.floor(math.log2(self.buffers[0].width))
        num_mip_levels = min(num_mip_levels, math.floor(math.log2(self.buffers[0].height)))

        width: int = self.buffers[0].width
        height: int = self.buffers[0].height
        buffer: Buffer = self.buffers[0]
        for _ in range(0, num_mip_levels):
            width = width // 2
            height = height // 2
            
            new_mipmap: Buffer = self.generate_mipmap(buffer, width, height, self.min_filtering_method)
            
            buffer = new_mipmap
            self.buffers.append(new_mipmap)

        return self

    def generate_mipmap(self, src: Buffer, target_width: int, target_height: int, method: FilterMethod) -> Buffer:
        data: list[float | Vec4] = []

        for y in range(0, target_height):
            for x in range(0, target_width):
                u: float = x / (max(target_width - 1, 1))
                v: float = y / (max(target_height - 1, 1))
                data.append(filter(src, u, v, method))

        mipmap: Buffer = Buffer(
            data=data, width=target_width, height=target_height,
            n_samples_per_axis=src.n_samples_per_axis, 
            format=src.format, color_space=src.color_space
        )
    
        return mipmap


@dataclass
class Sampler3D:
    cubemap: Cubemap

    min_filtering_method: FilterMethod = FilterMethod.BILINEAR
    mag_filtering_method: FilterMethod = FilterMethod.NEAREST
    
    def sample(self, dir: Vec3) -> Vec4:
        uv, face = dir_to_uv(dir)
        return sample2D(self.cubemap.faces[face], *uv, method=self.min_filtering_method)


def sample2D(buf: Buffer, u: float, v: float, method: FilterMethod, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
    if (buf.format == Format.D_UNORM or buf.format == Format.D_SFLOAT):
        border_color = [border_color, 0, 0, 1]

    if (mode == WrappingMode.CLAMP):
        u = max(min(u, 1), 0)
        v = max(min(v, 1), 0)
    elif (mode == WrappingMode.CLAMP_TO_BORDER):
        assert border_color != None, "no border color selected on border wrapping mode"
        if (u > 1 or u < 0 or v > 1 or v < 0):
            return Vec4(*border_color)
    elif (mode == WrappingMode.REPEAT):
        u = u % 1
        v = v % 1
        
    assert u >= 0 and u <= 1 and v >= 0 and v <= 1
    val: Any = filter(buf, u, v, method)
    
    if (buf.format == Format.D_UNORM or buf.format == Format.D_SFLOAT):
        val = [val, 0, 0, 1]

    val = transfer_format(val, buf.format, Format.RGBA_SFLOAT)
    val = transfer_color_space(val, buf.color_space, ColorSpace.LINEAR)

    return Vec4(*val)


def filter(buf: Buffer, u: float, v: float, method: FilterMethod) -> Any:
    n_samples: int = buf.n_samples_per_axis ** 2

    max_x: float = buf.width - 1
    max_y: float = buf.height - 1

    x: float = u * max_x
    y: float = v * max_y
    
    if (method == FilterMethod.NEAREST):
        return buf.data[(round(y) * buf.width + round(x)) * n_samples]
    elif (method == FilterMethod.BILINEAR):
        c0: Any = buf.data[(math.floor(y) * buf.width + math.floor(x)) * n_samples]
        c1: Any = buf.data[(math.ceil(y) * buf.width + math.floor(x)) * n_samples]
        c2: Any = buf.data[(math.floor(y) * buf.width + math.ceil(x)) * n_samples]
        c3: Any = buf.data[(math.ceil(y) * buf.width + math.ceil(x)) * n_samples]

        a: float = x % 1
        b: float = y % 1

        l0: Any = c0 + (c2 - c0) * a
        l1: Any = c1 + (c3 - c1) * a
        
        return l0 + (l1 - l0) * b

    assert False, "unsupported filtering method"
