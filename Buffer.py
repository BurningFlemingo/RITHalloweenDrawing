from typing import NamedTuple
from typing import Any
from VectorMath import *


class Buffer(NamedTuple):
    data: list[Any]
    width: int
    height: int
    n_samples_per_axis: int
    srgb_nonlinear: bool # color space of what is stored in the buffer

    # writes to all samples
    def write2D(self, x: int, y: int, val: Vec4):
        samples: int = self.n_samples_per_axis ** 2
        for sample in range(0, samples):
            self.data[samples * (y * self.width + x) + sample] = val

    def sampleUV(self, u: float, v: float):
        """
            u and v should be normalized between 0 and 1.
        """
        n_samples: int = self.n_samples_per_axis ** 2

        x: int = int(u * (self.width - 1))
        y: int = int(v * (self.height - 1))

        index: int = (y * self.width + x) * n_samples

        return self.data[index]


class Framebuffer(NamedTuple):
    color_attachment: Buffer
    resolve_attachment: Buffer
    depth_attachment: Buffer


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
            gamma: float = 2.2
            if (target.srgb_nonlinear and not src.srgb_nonlinear):
                final_color = final_color ** (1/gamma)
                
            elif(not target.srgb_nonlinear and src.srgb_nonlinear):
                final_color = final_color ** gamma
                
            target.data[target_px_index] = final_color 


def clear_buffer(buffer: Buffer, clear_value: Any) -> None:
    for i in range(0, len(buffer.data)):
        buffer.data[i] = clear_value
