from typing import NamedTuple
from typing import Callable
from dataclasses import dataclass

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


@dataclass
class Attachment:
    buffer: Buffer
    size_mode: SizeMode
    
    is_transient: bool
    is_allocated: bool


class AttachmentType(Enum):
    COLOR = 1, 
    DEPTH = 2


class AttachmentHandle(NamedTuple):
    uid: int
    type: AttachmentType
    
    debug_name: str = ""


class RenderPass:
    def __init__(self, viewport: Viewport, callback: Callable[[RenderCtx], None]):
        self.viewport = viewport

        self.input_attachments: list[AttachmentHandle] = []
        self.output_color_attachments: list[AttachmentHandle] = []
        
        self.depth_attachment: AttachmentHandle | None = None
        
        self.clear_values: dict[AttachmentHandle, Vec4 |
                                float | None] = {}  # [attachment.uid, ClearValue]

        self.render_callback: Callable[[RenderCtx], None] = callback

    def add_input_attachment(self, handle: AttachmentHandle):
        self.input_attachments.append(handle)

    def add_color_output(self, handle: AttachmentHandle, clear_color: Vec4 | None = None):
        assert handle not in self.output_color_attachments
        assert handle.type != AttachmentType.DEPTH, "depth attachment cannot be a color target"
        
        self.output_color_attachments.append(handle)
        self.clear_values[handle] = clear_color

    def set_depth_attachment(self, handle: AttachmentHandle, clear_value: float | None = None):
        assert handle.type == AttachmentType.DEPTH
        
        self.depth_attachment = handle
        self.clear_values[handle] = clear_value


class RenderGraph:
    def __init__(self):
        self.render_passes: list[RenderPass] = []
        self.attachments: list[Attachment] = []

    def make_attachment(self, info: AttachmentInfo, debug_name: str = "") -> AttachmentHandle:
        index: int = len(self.attachments)
        attachment_type: AttachmentType = AttachmentType.COLOR
        if (info.format == Format.D_UNORM or info.format == Format.D_SFLOAT):
            attachment_type = AttachmentType.DEPTH

        handle = AttachmentHandle(uid=index, type=attachment_type, debug_name=debug_name)
        buffer: Buffer = Buffer(data = [], width=info.width, height=info.height, n_samples_per_axis=info.msaa, format=info.format, color_space=info.color_space)
        attachment = Attachment(buffer=buffer, size_mode=info.size_mode, is_transient=info.is_transient, is_allocated=False)
        
        self.attachments.append(attachment)
        
        return handle

    def import_attachment(self, buffer: Buffer, debug_name: str ="") -> AttachmentHandle:
        index: int = len(self.attachments)
        attachment_type: AttachmentType = AttachmentType.COLOR
        if (buffer.format == Format.D_UNORM or buffer.format == Format.D_SFLOAT):
            attachment_type = AttachmentType.DEPTH
            
        handle = AttachmentHandle(uid=index, type=attachment_type, debug_name=debug_name)
        attachment = Attachment(buffer=buffer, size_mode=SizeMode.ABSOLUTE, is_transient=True, is_allocated=True)
            
        self.attachments.append(attachment)

        return handle
        

    def make_pass(self, viewport: Viewport, callback: Callable[[RenderCtx], None]) -> RenderPass:
        self.render_passes.append(RenderPass(viewport, callback))
        return self.render_passes[-1]

    def add_pass(self, render_pass: RenderPass):
        self.render_passes.append(render_pass)
        return self.render_passes[-1]

    def compile(self):
        for render_pass in self.render_passes:
            attachments: list[AttachmentHandle] | None = None
            if (render_pass.depth_attachment != None):
                attachments = render_pass.input_attachments + render_pass.output_color_attachments + [render_pass.depth_attachment]
            else:
                attachments = render_pass.input_attachments + render_pass.output_color_attachments
                
            for handle in attachments:
                assert handle.uid < len(self.attachments),\
                f"{handle.uid}:{handle.debug_name} attachment was never added to graph"
                
                attachment: Attachment = self.attachments[handle.uid]
                if (attachment.is_allocated == False):
                    if (attachment.size_mode == SizeMode.VIEWPORT):
                        attachment.buffer = Buffer(
                            data=attachment.buffer.data, 
                            width=render_pass.viewport.width, 
                            height=render_pass.viewport.height, 
                            n_samples_per_axis=attachment.buffer.n_samples_per_axis,
                            format=attachment.buffer.format,
                            color_space=attachment.buffer.color_space
                        )
                    allocate_buffer(attachment.buffer)
                    attachment.is_allocated = True

    def execute(self):
        for render_pass in self.render_passes:
            input_attachments: list[Buffer] = []
            color_output_attachments: list[Buffer] = []
            depth_attachment: Buffer | None = None

            msaa: int = 0

            if (render_pass.depth_attachment != None):
                uid: int = render_pass.depth_attachment.uid
                depth_attachment = self.attachments[uid].buffer
                msaa = depth_attachment.n_samples_per_axis
            
                
            for handle in render_pass.output_color_attachments:
                assert handle.uid < len(self.attachments),\
                f"{handle.uid}:{handle.debug_name} attachment not found in graph"

                attachment: Attachment = self.attachments[handle.uid]
                assert attachment.buffer != None
                
                buffer: Buffer = attachment.buffer
                
                clear_value: Vec4 | float | None = render_pass.clear_values[handle]
                if (clear_value != None):
                    clear_buffer(attachment.buffer, clear_value)

                msaa = max(msaa, buffer.n_samples_per_axis)
                
                color_output_attachments.append(buffer)

            for handle in render_pass.input_attachments:
                assert handle.uid < len(self.attachments),\
                f"{handle.uid}:{handle.debug_name} attachment not found in graph"

                attachment: Attachment = self.attachments[handle.uid]
                assert attachment.buffer != None

                buffer: Buffer = attachment.buffer
                input_attachments.append(buffer)

            viewport: Viewport = render_pass.viewport
            framebuffer: Framebuffer = Framebuffer(
                color_attachments=color_output_attachments,
                depth_attachment=depth_attachment,
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
    assert info.width != 0 and info.height != 0, "attachment size is zero"

    buffer: Buffer = Buffer(
        data=[], width=info.width, height=info.height,
        n_samples_per_axis=info.msaa,
        format=info.format, color_space=info.color_space
    )

    allocate_buffer(buffer)
    
    return buffer


def allocate_buffer(buf: Buffer) -> None:
    assert len(buf.data) == 0, "attempting to allocate already allocated buffer"
    
    clear_value: float | Vec4 | None = None
    if (buf.format == Format.RGBA_UNORM or buf.format == Format.RGBA_SFLOAT):
        clear_value = Vec4(0.0, 0.0, 0.0, 0.0)
    else:
        clear_value = 1.0

    n_samples: int = buf.n_samples_per_axis ** 2
    data: list[Vec4 | float] = [clear_value for _ in range(
        0, buf.width * buf.height * n_samples)]

    buf.data.extend(data)
