from typing import NamedTuple

from Rasterizer import *
from VectorMath import *
from MatrixMath import *
from Buffer import *
from Presentation import Viewport


VertexShader = Callable[[Any], Vertex]


class ShaderProgram(NamedTuple):
    vertex_shader: VertexShader
    fragment_shader: FragmentShader


def perspective_divide(vec: Vec4) -> Vec4:
    """
        w component is saved.
    """
    return Vec4(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w, vec.w)


def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4:
    x_px: float = ((ndc.x + 1.0) * 0.5) * width
    y_px: float = ((ndc.y + 1.0) * 0.5) * height
    return Vec4(x_px, y_px, ndc.z, ndc.w)


def draw(framebuffer: Framebuffer, viewport: Viewport, program: ShaderProgram, vertex_buffers: dict[str, list], first: int, count: int):
    vertices: list[Vertex] = []
    attribute_fields: list[str] = program.vertex_shader.Attributes._fields
    for i in range(first, first + count):
        attribute_values: list = [vertex_buffers[attribute][i]
                                  for attribute in attribute_fields]

        vertex_attributes = program.vertex_shader.Attributes(*attribute_values)

        shaded_vertex: Vertex = program.vertex_shader(vertex_attributes)

        ndc_pos = perspective_divide(shaded_vertex.pos)
        screen_space_pos = viewport_transform(
            ndc_pos, viewport.width, viewport.height)

        vertex: Vertex = Vertex(
            pos=screen_space_pos, fragment_attributes=shaded_vertex.fragment_attributes)
        vertices.append(vertex)

    for i in range(0, len(vertices), 3):
        v1: Vertex = vertices[i]
        v2: Vertex = vertices[i + 1]
        v3: Vertex = vertices[i + 2]

        rasterize_triangle(
            framebuffer, program.fragment_shader, v1, v2, v3)
