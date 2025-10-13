from typing import NamedTuple
from typing import Any
from enum import Enum
from VectorMath import *


class WrappingMode(Enum):
    NONE = 1,
    CLAMP = 2,
    CLAMP_TO_BORDER = 3,
    REPEAT = 4


class Format(Enum):
    UNORM = 1
    SFLOAT = 2


class ColorSpace(Enum):
    LINEAR = 1
    SRGB = 2


class Buffer(NamedTuple):
    data: list[NamedTuple]
    width: int
    height: int
    n_samples_per_axis: int

    format: Format
    color_space: ColorSpace

    def write_samples(self, x: int, y: int, val: NamedTuple, sample_indices: list[int]):
        val = transfer_color_space(val, ColorSpace.LINEAR, self.color_space)
        val = transfer_format(val, Format.SFLOAT, self.format)

        samples: int = self.n_samples_per_axis ** 2
        for sample in sample_indices:
            self.data[samples * (y * self.width + x) + sample] = val

    def sampleUV(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
        """
            u and v should be normalized between 0 and 1.
        """
        if (type(border_color) != tuple and type(border_color) != list and type(border_color) != None):
            border_color = [border_color]

        n_samples: int = self.n_samples_per_axis ** 2

        max_x: int = self.width - 1
        max_y: int = self.height - 1

        x: int = int(u * max_x)
        y: int = int(v * max_y)

        if (mode == WrappingMode.CLAMP):
            x = max(min(x, max_x), 0)
            y = max(min(y, max_y), 0)
        elif (mode == WrappingMode.CLAMP_TO_BORDER):
            assert type(
                border_color) is not None, "no border color selected on border wrapping mode"
            if (x > max_x or x < 0 or y > max_y or y < 0):
                return Vec4(*border_color)
        elif (mode == WrappingMode.REPEAT):
            x = x % max_x
            y = y % max_y

        index: int = (y * self.width + x) * n_samples

        val: NamedTuple = self.data[index]
        if (type(val) != tuple and type(val) != list):
            val = [val]
        val = transfer_color_space(val, self.color_space, ColorSpace.LINEAR)
        val = transfer_format(val, self.format, Format.SFLOAT)

        return Vec4(*val)


def transfer_format(src_val: NamedTuple, src_format: Format, dst_format: Format) -> NamedTuple:
    if (src_format == Format.SFLOAT and dst_format == Format.UNORM):
        elements: list[float] = []
        for element in src_val:
            elements.append(max(min(element, 1.0), 0.0))
        return type(src_val)(*elements)
    return src_val

def transfer_color_space(src_color: NamedTuple, src_space: ColorSpace, dst_space: ColorSpace) -> NamedTuple:
    gamma: float = 2.2
    if (dst_space == ColorSpace.SRGB and src_space == ColorSpace.LINEAR):
        channels: list[float] = []
        for channel in src_color[:3]:
            channels.append(channel ** (1/gamma))
        channels += src_color[3:]
        return type(src_color)(*channels)

    elif (dst_space == ColorSpace.LINEAR and src_color == ColorSpace.SRGB):
        channels: list[float] = []
        for channel in src_color[:3]:
            channels.append(channel ** gamma)
        channels += src_color[3:]
        return type(src_color)(*channels)
    return src_color


class Framebuffer(NamedTuple):
    color_attachments: list[Buffer]
    resolve_attachments: list[Buffer]
    depth_attachment: Buffer

    width: int
    height: int

    n_samples_per_axis: int


def resolve_buffer(src: Buffer, target: Buffer):
    """
        src buffer must be the same width and height as the target buffer. 
    """
    n_samples: int = src.n_samples_per_axis ** 2
    for y_px in range(0, src.height):
        for x_px in range(0, src.width):
            target_px_index: int = y_px * target.width + x_px
            src_px_index: int = target_px_index * n_samples

            accumulated_value = src.data[src_px_index]
            for sample_index in range(1, n_samples):
                accumulated_value = accumulated_value + \
                    src.data[src_px_index + sample_index]

            avg_color: Vec4 = accumulated_value * (1/n_samples)
            avg_color = transfer_color_space(avg_color, src.color_space, target.color_space)
            final_color = transfer_format(avg_color, src.format, target.format)

            target.data[target_px_index] = final_color


def clear_buffer(buffer: Buffer, clear_value: Any) -> None:
    for i in range(0, len(buffer.data)):
        buffer.data[i] = clear_value
