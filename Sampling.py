from typing import NamedTuple
from typing import Any
from enum import Enum
from VectorMath import *
from Buffer import *
from Cubemap import *

class FilterMethod(Enum):
    NEAREST = 0,
    BILINEAR = 1,


class SizeDimensions(NamedTuple):
    width: int
    height: int

class Sampler2D(NamedTuple):
    base: Buffer
    mip_chain: list[Buffer] = []

    min_filtering_method: FilterMethod = FilterMethod.BILINEAR
    mag_filtering_method: FilterMethod = FilterMethod.NEAREST
    
    dudx: float = 0
    dudy: float = 0
    dvdx: float = 0
    dvdy: float = 0

    def get_size(self, lod: int=0) -> SizeDimensions:
        return SizeDimensions(self.base.width, self.base.height)
    
    def sample(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
        return sample2D(self.base, u, v, self.min_filtering_method, mode, border_color)

    def generate_mipmaps(self) -> Sampler2D:
        if (len(self.mip_chain) > 0):
            return self
        
        num_mip_levels: int = math.floor(math.log2(self.base.width))
        num_mip_levels = min(num_mip_levels, math.floor(math.log2(self.base.height)))

        width: int = self.base.width
        height: int = self.base.height
        buffer: Buffer = self.base
        for _ in range(0, num_mip_levels):
            width = width // 2
            height = height // 2
            
            new_mipmap: Buffer = self.generate_mipmap(buffer, width, height, self.min_filtering_method)
            
            buffer = new_mipmap
            self.mip_chain.append(new_mipmap)

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



class Sampler3D(NamedTuple):
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
