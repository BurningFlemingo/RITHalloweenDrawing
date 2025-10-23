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
    RGBA_UNORM = 1,
    RGBA_SFLOAT = 2,
    D_UNORM = 3,
    D_SFLOAT = 4,


class ColorSpace(Enum):
    NONE = 0
    LINEAR = 1
    SRGB = 2


class Buffer(NamedTuple):
    data: list[Vec4 | float]
    
    width: int
    height: int

    format: Format
    color_space: ColorSpace = ColorSpace.NONE
    
    n_samples_per_axis: int = 1
    
    def write_samples(self, x: int, y: int, val: Any, sample_indices: list[int]):
        val = transfer_format(val, Format.RGBA_SFLOAT, self.format)
        val = transfer_color_space(val, ColorSpace.LINEAR, self.color_space)

        samples: int = self.n_samples_per_axis ** 2
        for sample in sample_indices:
            self.data[samples * (y * self.width + x) + sample] = val


class Framebuffer(NamedTuple):
    color_attachments: list[Buffer]
    depth_attachment: Buffer | None

    width: int
    height: int

    n_samples_per_axis: int


def transfer_format(src_val: Any, src_format: Format, dst_format: Format) -> Vec4 | float:
    if (src_format == Format.RGBA_SFLOAT and dst_format == Format.RGBA_UNORM):
        elements: list[float] = []
        for element in src_val:
            elements.append(max(min(element, 1.0), 0.0))
        return type(src_val)(*elements)
    return src_val


def transfer_color_space(src_color: Any, src_space: ColorSpace, dst_space: ColorSpace) -> NamedTuple:
    gamma: float = 2.2
    if (dst_space == ColorSpace.SRGB and src_space == ColorSpace.LINEAR):
        channels: list[float] = []
        for channel in src_color[:3]:
            channels.append(channel ** (1/gamma))
        channels += src_color[3:]
        return type(src_color)(*channels)

    elif (dst_space == ColorSpace.LINEAR and src_space == ColorSpace.SRGB):
        channels: list[float] = []
        for channel in src_color[:3]:
            channels.append(channel ** gamma)
        channels += src_color[3:]
        return type(src_color)(*channels)
    return src_color


def transfer_buffer(src: Buffer, target_format: Format, target_color_space: ColorSpace) -> Buffer:
    n_samples: int = src.n_samples_per_axis ** 2
    for y_px in range(0, src.height):
        for x_px in range(0, src.width):
            px_index: int = (y_px * src.width + x_px) * n_samples
            for sample_index in range(0, n_samples):
                index: int = px_index + sample_index
                value: Any = src.data[index]

                final_value = transfer_format(value, src.format, target_format)
                final_value = transfer_color_space(
                    final_value, src.color_space, target_color_space)

                src.data[index] = final_value

    return Buffer(
        data=src.data, width=src.width, height=src.height,
        n_samples_per_axis=src.n_samples_per_axis,
        format=target_format, color_space=target_color_space
    )


def resolve_buffer(src: Buffer, target: Buffer):
    """
        src buffer must be the same width and height as the target buffer. 
    """
    assert (src.width == target.width)
    assert (src.height == target.height)

    n_samples: int = src.n_samples_per_axis ** 2
    for y_px in range(0, src.height):
        for x_px in range(0, src.width):
            target_px_index: int = y_px * target.width + x_px
            src_px_index: int = target_px_index * n_samples

            accumulated_value = src.data[src_px_index]
            for sample_index in range(1, n_samples):
                accumulated_value = accumulated_value + \
                    src.data[src_px_index + sample_index]

            avg_color: Vec4 = accumulated_value / n_samples

            final_color = transfer_format(avg_color, src.format, target.format)
            final_color = transfer_color_space(
                final_color, src.color_space, target.color_space)

            target.data[target_px_index] = final_color

# def transition_buffer(buffer: Buffer, dst_format: Format, dst_color_space: ColorSpace) -> Buffer:


def clear_buffer(buffer: Buffer, clear_value: Any) -> None:
    for i in range(0, len(buffer.data)):
        buffer.data[i] = clear_value
