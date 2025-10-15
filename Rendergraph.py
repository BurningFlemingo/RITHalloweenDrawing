from typing import NamedTuple
from typing import Callable
from Buffer import *
from enum import IntEnum

class AttachmentType(IntEnum):
    COLOR = 0
    DEPTH = 1

class AttachmentDescription(NamedTuple):
    attachment_type: AttachmentType
    width: int 
    height: int 
    msaa: int = 1 # how many samples per axis
    
    format: Format = Format.SFLOAT
    color_space: ColorSpace = ColorSpace.LINEAR
    
    clear_value: Any | None = None

class RenderCtx:
    attachments: list[AttachmentDescription]

class RenderGraph:
    def __init__(self):
        pass
        
    def add_resource(self, name: str, description: AttachmentDescription, type: AttachmentType):
        self.attachments[attachment_type]
        
    def add_render_pass(self, render_callback: Callable[[RenderCtx], None]):

