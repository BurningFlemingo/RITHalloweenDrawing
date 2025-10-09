from typing import NamedTuple

class Buffer(NamedTuple):
    data: list[Any]
    width: int
    height: int
    n_samples_per_axis: int

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
    depth_attachment: Buffer
    

def resolve_buffer(buffer: Buffer) -> None:
    n_samples: int = buffer.n_samples_per_axis ** 2
    for j in range(0, buffer.height):
        for i in range(0, buffer.width):
            px_index: int = (j * buffer.width + i) * n_samples
            accumulated_value = buffer.data[px_index]
            for sample_index in range(1, n_samples):
                accumulated_value = accumulated_value + \
                    buffer.data[px_index + sample_index]
            average_value = accumulated_value * (1/n_samples)
            for sample_index in range(0, n_samples):
                buffer.data[px_index + sample_index] = average_value


def clear_buffer(buffer: Buffer, clear_value: Any) -> None:
    for i in range(0, len(buffer.data)):
        buffer.data[i] = clear_value
