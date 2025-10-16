from typing import NamedTuple
from typing import Callable
from enum import IntEnum, Enum
from dataclasses import dataclass, field

from Buffer import *
from Presentation import Viewport
from Rasterizer import FragmentShader
from Renderer import *
from VectorMath import *


class RenderCtx(NamedTuple):
    def draw(self, vertex_buffer: dict[str, list], vertex_shader: VertexShader, fragment_shader: FragmentShader, vertex_count: int, vertex_offset: int = 0):
        draw(self.framebuffer, self.viewport, vertex_buffer,
             vertex_shader, fragment_shader, vertex_count, vertex_offset)

    input_attachments: dict[str, Buffer]
    framebuffer: Framebuffer
    viewport: Viewport


class SizeMode(Enum):
    DEFAULT = 0,  # ignores width and height
    ABSOLUTE = 1,


@dataclass
class AttachmentInfo:
    size_mode: SizeMode = SizeMode.BACKBUFFER

    width: int = 0
    height: int = 0

    msaa: int = 1

    format: Format = Format.SFLOAT
    color_space: ColorSpace = ColorSpace.LINEAR

    is_transient: bool = True


class RenderPass:
    def __init__(self, callback: Callable[[RenderCtx], None]):
        self.inputs: list[str] = []

        self.color_outputs: list[str] = []
        self.depth_output: str | None = None

        self.clear_values: dict[str, Vec4 |
                                float | None] = {}  # [str, ClearValue]

        self.render_callback: Callable[[RenderCtx], None] | None = callback

    def add_input_attachment(self, name: str) -> RenderPass:
        self.inputs.append(name)
        return self

    def add_color_output(self, name: str, clear_color: Vec4 | None = None) -> RenderPass:
        self.color_outputs.append(name)
        self.clear_values[name] = clear_color

        return self

    def set_depth_output(self, name: str, clear_value: float | None = None) -> RenderPass:
        self.depth_output = name
        self.clear_values[name] = clear_color

        return self


class RenderGraph:
    def __init__(self, default_width: int, default_height: int):
        self.render_passes: list[RenderPass] = []
        self.attachments: dict[str, Buffer] = {}
        self.attachment_infos: dict[str, AttachmentInfo] = {}

        self.default_width = default_width
        self.default_height = default_height

    def add_resource(self, name: str, info: AttachmentInfo):
        self.attachment_infos[name] = info

    def add_pass(self, render_pass: RenderPass) -> RenderPass:
        self.render_passes.append(render_pass)
        return self.render_passes[-1]

    def set_backbuffer(self, name: str, backbuffer: Buffer):
        assert (backbuffer.color_space == ColorSpace.SRGB)
        assert (backbuffer.format == Format.UNORM)
        assert (backbuffer.n_samples_per_axis == 1)

        self.backbuffer = backbuffer
        self.backbuffer_name = name

    def get_attachment(self, name: str) -> Buffer:
        return self.attachments[name]

    def compile(self):
        assert (self.backbuffer != None and self.backbuffer_name != "")

        for render_pass in self.render_passes:
            for name in render_pass.color_outputs():
                assert name in self.attachments, "attachment referenced that was never added to graph"

                info: AttachmentInfo = self.attachments[name]
                if (name not in self.attachments):
                    if (info.size_mode == SizeMode.BACKBUFFER):
                        info.width = self.backbuffer.width
                        info.height = self.backbuffer.height

                    if (name != render_pass.depth_output):
                        self.attachments[name] = make_buffer(
                            info, Vec4(0, 0, 0, 0))
                    else:
                        self.attachments[name] = make_buffer(
                            info, float("inf"))

    def execute(self):
        for render_pass in self.render_passes:
            if (len(render_pass.attachments) == 0):
                continue

            input_attachments: dict[str, Buffer] = {}
            color_output_attachments: list[Buffer] = []
            depth_output_attachment: Buffer | None = None

            max_width: int = 0
            max_height: int = 0

            for name in render_pass.inputs:
                buffer: Buffer = self.attachments[name]
                info: AttachmentInfo = render_pass.attachments[name]
                if (info.clears[name]):
                    clear_buffer(attachment_buffer, info.clear_value)

                if (info.msaa != buffer.n_samples_per_axis):
                    new_buffer: Buffer = make_buffer(info)
                    resolve_buffer(buffer, new_buffer)
                    buffer = new_buffer
                elif (info.format != buffer.format or info.color_space != buffer.color_space):
                    buffer = transfer_buffer(
                        buffer, info.format, info.color_space)

                input_attachments[name] = buffer

            ctx: RenderCtx = RenderCtx()
            for input in render_pass.input:
                if (input not in self.attachment_buffers):
                    self.attachment_buffers[input] = make_buffer(
                        self.attachments[input])
                ctx.input[input] = self.attachment_buffers[input]
            for output in render_pass.input:
                if (output not in self.attachment_buffers):
                    self.attachment_buffers[output] = make_buffer(
                        self.attachments[output])

            render_pass.render_callback(ctx)


def make_buffer(info: AttachmentInfo, clear_value: Any) -> Buffer:
    data: list[NamedTuple] = [clear_value for _ in range(
        0, info.width * info.height * info.msaa)]
    buffer: Buffer = Buffer(
        data=data, width=info.width, height=info.height,
        n_samples_per_axis=info.msaa,
        format=info.format, color_space=info.color_space
    )
    return buffer
