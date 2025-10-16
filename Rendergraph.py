from typing import NamedTuple
from typing import Callable
from enum import IntEnum, Enum
from dataclasses import dataclass, field

from Buffer import *
from Presentation import Viewport
from Rasterizer import FragmentShader
from Renderer import *


class RenderCtx(NamedTuple):
    def draw(self, vertex_buffer: dict[str, list], vertex_shader: VertexShader, fragment_shader: FragmentShader, vertex_count: int, vertex_offset: int = 0):
        draw(self.framebuffer, self.viewport, vertex_buffer, vertex_shader, fragment_shader, vertex_count, vertex_offset)

    input_attachments: dict[str, Buffer]
    framebuffer: Framebuffer
    viewport: Viewport


class Usage(Enum):
    READ_ONLY = 0,
    WRITE_ONLY = 1

class SizeMode(Enum):
    BACKBUFFER = 0, # ignores width and height
    ABSOLUTE = 1, 


@dataclass
class AttachmentInfo:
    clear_value: Vec4 | float
    
    size_mode: SizeMode = SizeMode.BACKBUFFER
    width: int = 0
    height: int = 0
    
    msaa: int = 1
    
    format: Format = Format.SFLOAT
    color_space: ColorSpace = ColorSpace.LINEAR


class RenderPass:
    def __init__(self):
        self.attachments: dict[str, AttachmentInfo] = {}
        
        self.inputs: list[str] = []
        self.color_outputs: list[str] = []
        
        self.depth_output: str | None = None
        
        self.clears: dict[str, Vec4 | float] = {} # [str, ClearValue]
        
        self.render_callback: Callable[[RenderCtx], None] | None = None

    def add_input_attachment(self, name: str) -> RenderPass:
        self.inputs.append(name)
        return self


    def add_color_output(self, name: str, info: AttachmentInfo, clear_color: Vec4 | None = None) -> RenderPass:
        self.attachments[name] = info
        self.color_outputs.append(name)
        if (clear_color != None):
            self.input_clear[name] = clear_color 

        return self

    def set_depth_output(self, name: str, info: AttachmentInfo, clear_value: float | None = None) -> RenderPass:
        """
            Most recent depth write attachment will be used.
        """
        self.attachments[name] = info
        self.depth_output = name
        if (clear_value != None):
            self.input_clear[name] = clear_value

        return self


    def set_render_callback(self, callback: Callable[[RenderCtx], None]) -> RenderPass:
        self.render_callback = callback

        return self
            

class RenderGraph:
    def __init__(self):
        self.render_passes: list[RenderPass] = []
        self.attachments: dict[str, Buffer] = {}
        
        self.backbuffer: Buffer | None = None
        self.backbuffer_name = ""

    def add_resource(self, name: str, info: AttachmentInfo):
        if (info.width == None or info.height == None):
            info.width = self.backbuffer.width
            info.height = self.backbuffer.height
            
        self.attachments[name] = make_buffer(info)

    def add_pass(self, render_pass: RenderPass) -> RenderPass:
        self.render_passes.append(render_pass)
        return self.render_passes[-1]

    def set_backbuffer(self, name: str, backbuffer: Buffer):
        assert(backbuffer.color_space == ColorSpace.SRGB)
        assert(backbuffer.format == Format.UNORM)
        assert(backbuffer.n_samples_per_axis == 1)

        self.backbuffer = backbuffer
        self.backbuffer_name = name
        
    def get_attachment(self, name: str) -> Buffer:
        return self.attachments[name]

    def compile(self):
        assert(self.backbuffer != None and self.backbuffer_name != "")
        
        for render_pass in self.render_passes:
            for name, info in render_pass.attachments.items():
                if (name not in self.attachments):
                    if (info.size_mode == SizeMode.BACKBUFFER):
                        info.width = self.backbuffer.width
                        info.height = self.backbuffer.height
                        
                    self.attachments[name] = make_buffer(info)
                    

    def execute(self):
        for render_pass in self.render_passes:
            input_attachments: dict[str, Buffer] = {}
            color_output_attachments: list[Buffer]
            framebuffer: Framebuffer | None = None
            viewport: Viewport | None = None
            
            for name in render_pass.inputs:
                buffer: Buffer = self.attachments[name]
                info: AttachmentInfo = render_pass.attachments[name]
                if (info.input_clear[name]):
                    clear_buffer(attachment_buffer, info.clear_value)
                    
                if (info.msaa != buffer.n_samples_per_axis):
                    new_buffer: Buffer = make_buffer(info)
                    resolve_buffer(buffer, new_buffer)
                    buffer = new_buffer
                elif (info.format != buffer.format or info.color_space != buffer.color_space):
                    buffer = transfer_buffer(buffer, info.format, info.color_space)
                    
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


def make_buffer(info: AttachmentInfo) -> Buffer:
    data: list[NamedTuple] = [info.clear_value for _ in range(
        0, info.width * info.height * info.msaa)]
    buffer: Buffer = Buffer(
        data=data, width=info.width, height=info.height,
        n_samples_per_axis=info.msaa,
        format=info.format, color_space=info.color_space
    )
    return buffer
