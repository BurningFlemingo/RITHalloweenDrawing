from typing import NamedTuple

from Rasterizer import *
from VectorMath import *
from MatrixMath import *
from Buffer import *
from Presentation import Viewport
from Clipping import *


VertexShader = Callable[[Any], Vertex]


class GraphicsPipeline(NamedTuple):
    viewport: Viewport
    framebuffer: Framebuffer


def draw(pipeline: GraphicsPipeline, vertex_buffer: dict[str, list], vertex_shader: VertexShader, fragment_shader: FragmentShader | None, vertex_count: int, vertex_offset: int):
    vertices: list[Vertex] = []
    attribute_fields: list[str] = vertex_shader.Attributes._fields
    for i in range(vertex_offset, vertex_count + vertex_offset):
        attribute_values: list = [vertex_buffer[attribute][i]
                                  for attribute in attribute_fields]

        vertex_attributes = vertex_shader.Attributes(
            *attribute_values)

        shaded_vertex: Vertex = vertex_shader(vertex_attributes)
        vertices.append(shaded_vertex)

    for i in range(0, len(vertices), 3):

        v1: Vertex = vertices[i]
        v2: Vertex = vertices[i + 1]
        v3: Vertex = vertices[i + 2]

        triangles: list[Triangles] = clip_triangle(v1, v2, v3)
        for triangle in triangles:
            v1_px: Vertex = viewport_transform(pipeline.viewport, triangle.v1)
            v2_px: Vertex = viewport_transform(pipeline.viewport, triangle.v2)
            v3_px: Vertex = viewport_transform(pipeline.viewport, triangle.v3)
            rasterize_triangle(
                pipeline.framebuffer, fragment_shader, v1_px, v2_px, v3_px)


def viewport_transform(viewport: Viewport, ndc: Vertex) -> Vertex:
    x_px: float = ((ndc.pos.x + 1.0) * 0.5) * (viewport.width - 1)
    y_px: float = ((ndc.pos.y + 1.0) * 0.5) * (viewport.height - 1)
    new_pos: Vec4 = Vec4(x_px, y_px, ndc.pos.z, ndc.pos.w)
    return Vertex(pos=new_pos, fragment_attributes=ndc.fragment_attributes)
