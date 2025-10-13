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
    srgb_nonlinear: bool  # color space of what is stored in the buffer

    format: Format
    color_space: ColorSpace

    # writes to all samples
    def write2D(self, x: int, y: int, val: NamedTuple):
        val = transfer_color_space(val, ColorSpace.LINEAR, self.color_space)
        samples: int = self.n_samples_per_axis ** 2
        for sample in range(0, samples):
            self.data[samples * (y * self.width + x) + sample] = val

    def sampleUV(self, u: float, v: float, mode: WrappingMode = WrappingMode.NONE, border_color: Any | None = None):
        """
            u and v should be normalized between 0 and 1.
        """
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
                return border_color
        elif (mode == WrappingMode.REPEAT):
            x = x % max_x
            y = y % max_y

        index: int = (y * self.width + x) * n_samples

        val: Any = self.data[index]


def transfer_color_space(src_color: Vec3, src_space: ColorSpace, dst_space: ColorSpace):
    gamma: float = 2.2
    if (dst_space == SRGB and src_space != SRGB):
        return src_color ** (1/gamma)

    elif (dst_space != SRGB and src_color == SRGB):
        return src_color ** gamma


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

            average_color: Vec4 = accumulated_value * (1/n_samples)
            final_color: Vec3 = Vec3(*average_color[:3])

            target.data[target_px_index] = final_color


def clear_buffer(buffer: Buffer, clear_value: Any) -> None:
    for i in range(0, len(buffer.data)):
        buffer.data[i] = clear_value
