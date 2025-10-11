from typing import NamedTuple

from Rasterizer import *
from VectorMath import *
from MatrixMath import *
from Buffer import *
from Presentation import Viewport

from shaders.First_frag import *
from shaders.First_vert import *


class ShaderProgram(NamedTuple):
    vertex_shader: VertShaderObject
    fragment_shader: FragShaderObject


def perspective_divide(vec: Vec4) -> Vec4:
    """
        w component is saved.
    """
    return Vec4(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w, vec.w)


def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4:
    x_px: float = ((ndc.x + 1.0) * 0.5) * width
    y_px: float = ((ndc.y + 1.0) * 0.5) * height
    return Vec4(x_px, y_px, ndc.z, ndc.w)


def draw(framebuffer: Framebuffer, viewport: Viewport, program: ShaderProgram, vertex_buffers: tuple[list], first: int, count: int):
    vertices: list[Vertex] = []
    for i in range(first, count):
        vertex_attributes = program.vertex_shader.Attributes(
            *[buffer[i] for buffer in vertex_buffers])
        
        pos, fragment_attributes = program.vertex_shader(vertex_attributes)

        pos = perspective_divide(pos)
        pos = viewport_transform(
            pos, viewport.width, viewport.height)

        vertex: Vertex = Vertex(
            pos=pos, fragment_attributes=fragment_attributes)
        vertices.append(vertex)

    for i in range(0, len(vertices), 3):
        v1: Vertex = vertices[i]
        v2: Vertex = vertices[i + 1]
        v3: Vertex = vertices[i + 2]

        rasterize_triangle(
            framebuffer, program.fragment_shader, v1, v2, v3)
