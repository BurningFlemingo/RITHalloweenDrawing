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

    input_attachments: list[Buffer]
    framebuffer: Framebuffer
    viewport: Viewport


class SizeMode(Enum):
    VIEWPORT = 0,  # ignores width and height
    ABSOLUTE = 1,


@dataclass
class AttachmentInfo:
    size_mode: SizeMode = SizeMode.VIEWPORT

    width: int = 0
    height: int = 0

    msaa: int = 1

    format: Format = Format.RGBA_SFLOAT
    color_space: ColorSpace = ColorSpace.LINEAR

    is_transient: bool = True


class RenderPass:
    def __init__(self, viewport: Viewport, callback: Callable[[RenderCtx], None]):
        self.viewport = viewport
        
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
        self.clear_values[name] = clear_value

        return self


class RenderGraph:
    def __init__(self):
        self.render_passes: list[RenderPass] = []
        self.attachments: dict[str, Buffer] = {}
        self.attachment_infos: dict[str, AttachmentInfo] = {}

    def declare_attachment(self, name: str, info: AttachmentInfo):
        self.attachment_infos[name] = info

    def import_attachment(self, name: str, buffer: Buffer):
        attachment_info = AttachmentInfo(
            size_mode=SizeMode.ABSOLUTE, 
            width=buffer.width, 
            height=buffer.height,
            msaa=buffer.n_samples_per_axis,
            format=buffer.format,
            color_space=buffer.color_space,
            is_transient=False
        )
        self.attachments[name] = buffer
        self.attachment_infos[name] = attachment_info

    def add_pass(self, render_pass: RenderPass) -> RenderPass:
        self.render_passes.append(render_pass)
        return self.render_passes[-1]


    def get_attachment(self, name: str) -> Buffer:
        return self.attachments[name]

    def compile(self):
        for render_pass in self.render_passes:
            for name in render_pass.color_outputs:
                assert name in self.attachment_infos, f"{name} attachment was never added to graph"

                info: AttachmentInfo = self.attachment_infos[name]
                if (name not in self.attachments):
                    if (info.size_mode == SizeMode.VIEWPORT):
                        info.width = render_pass.viewport.width
                        info.height = render_pass.viewport.height

                    self.attachments[name] = make_buffer(info)

    def execute(self):
        for render_pass in self.render_passes:
            input_attachments: list[Buffer] = []
            color_output_attachments: list[Buffer] = []
            depth_output_attachment: Buffer | None = None
            
            msaa: int = 0
            
            assert render_pass.depth_output == None or render_pass.depth_output in self.attachments, f"{render_pass.depth_output} attachment not found in graph"
            if (render_pass.depth_output in self.attachments):
                depth_output_attachment: Buffer = self.attachments[render_pass.depth_output]
                
                msaa = max(msaa, depth_output_attachment.n_samples_per_axis)


            for name in render_pass.color_outputs:
                assert name in self.attachments, f"{name} attachment not found in graph"
                
                buffer: Buffer = self.attachments[name]
                if (info.clear_values[name] != None):
                    clear_buffer(buffer, info.clear_value)
                    
                color_output_attachments.append(buffer)
                
                msaa = max(msaa, buffer.n_samples_per_axis)

            for name in render_pass.inputs:
                assert name in self.attachments, f"{name} attachment not found in graph"
                
                buffer: Buffer = self.attachments[name]
                input_attachments.append(buffer)
                
            viewport: Viewport = render_pass.viewport
            framebuffer: Framebuffer = Framebuffer(
                color_attachments=color_output_attachments, 
                depth_attachment=depth_output_attachment, 
                n_samples_per_axis=msaa, 
                width=viewport.width, 
                height=viewport.height
            )

            ctx: RenderCtx = RenderCtx(
                input_attachments=input_attachments, 
                framebuffer=framebuffer, 
                viewport=viewport
            )
            
            render_pass.render_callback(ctx)


def make_buffer(info: AttachmentInfo) -> Buffer:
    clear_value: float | Vec4 | None = None
    if (info.format == Format.RGBA_UNORM or info.format == Format.RGBA_SFLOAT):
        clear_value = Vec4(0.0, 0.0, 0.0, 0.0)
    else:
        clear_value = 0.0

    assert info.width != 0 and info.height != 0, "attachment size is zero"
    if (info.width != 0 or info.height != 0):
        print("uh oh")
        
    data: list[NamedTuple] = [clear_value for _ in range(
        0, info.width * info.height * info.msaa)]
    buffer: Buffer = Buffer(
        data=data, width=info.width, height=info.height,
        n_samples_per_axis=info.msaa,
        format=info.format, color_space=info.color_space
    )
    return buffer
