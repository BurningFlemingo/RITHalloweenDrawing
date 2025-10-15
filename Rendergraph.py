from typing import NamedTuple
from typing import Callable
from enum import IntEnum
from dataclasses import dataclass, field

from Buffer import *
from Presentation import Viewport


class AttachmentType(IntEnum):
    COLOR = 0
    DEPTH = 1


class AttachmentDescription(NamedTuple):
    width: int
    height: int
    attachment_type: AttachmentType

    msaa: int = 1  # how many samples per axis

    format: Format = Format.SFLOAT
    color_space: ColorSpace = ColorSpace.LINEAR

    clear_value: Any | None = None


@dataclass
class RenderCtx:
    input: dict[str, Buffer]


class RenderPass(NamedTuple):
    inputs: list[str]
    outputs: list[str]
    render_callback: Callable[[RenderCtx], None]
    dependencies: list[RenderPass] = []


class RenderGraph:
    def __init__(self):
        self.render_passes: list[RenderPass] = []
        self.attachment_buffers[name]: dict[str, Buffer] = []

    def add_resource(self, name: str, description: AttachmentDescription) -> self:
        self.attachments[name]: dict[str, AttachmentDescription] = description
        self.attachment_buffers[name]: dict[str,
                                            AttachmentDescription] = description

    def add_render_pass(self, inputs: list[str], outputs: list[str], render_callback: Callable[[RenderCtx], None]):
        render_pass: RenderPass = RenderPass(
            inputs=inputs, outputs=outputs, render_callback=render_callback
        )
        self.render_passes.append(render_pass)

    def execute(self):
        for render_pass in self.render_passes:
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


def make_buffer(desc: AttachmentDescription) -> Buffer:
    data: list[NamedTuple] = [desc.clear_value for _ in range(
        0, desc.width * desc.height * desc.msaa)]
    buffer: Buffer = Buffer(
        data=data, width=desc.width, height=desc.height,
        n_samples_per_axis=desc.msaa,
        format=desc.format, color_space=desc.color_space
    )
    return buffer
