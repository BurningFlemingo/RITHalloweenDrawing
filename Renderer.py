from typing import NamedTuple

from Rasterizer import *
from VectorMath import *
from MatrixMath import *
from Buffer import *
from Presentation import Viewport


VertexShader = Callable[[Any], Vertex]


class GraphicsPipeline(NamedTuple):
    viewport: Viewport
    framebuffer: Framebuffer


class Triangle(NamedTuple):
    v1: Vertex
    v2: Vertex
    v3: Vertex


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


def clip_triangle(v1: Vertex, v2: Vertex, v3: Vertex) -> list[Triangle]:
    planes: list[Vec4] = [
        Vec4(0, 0, 1, 1),  # near
        Vec4(0, 0, -1, 1),  # far
        Vec4(1, 0, 0, 1),  # left
        Vec4(-1, 0, 0, 1),  # right
        Vec4(0, -1, 0, 1),  # top
        Vec4(0, 1, 0, 1)  # bottom
    ]

    triangles: list[Triangle] = [Triangle(v1, v2, v3)]

    centroid: Vec4 = (v1.pos + v2.pos + v3.pos) / 3

    v1_radius: float = (v1.pos - centroid).magnitude()
    v2_radius: float = (v2.pos - centroid).magnitude()
    v3_radius: float = (v3.pos - centroid).magnitude()

    bounding_sphere_radius: float = max(max(v1_radius, v2_radius), v3_radius)

    for plane in planes:
        sphere_distance: float = dot(plane, centroid)

        # reduce clip quality, but improve perf
        if (sphere_distance < -bounding_sphere_radius):
            return []
        # if (sphere_distance > bounding_sphere_radius):
        #     continue

        new_triangles: list[Triangle] = []
        for triangle in triangles:
            new_triangles += clip_triangle_against_plane(
                plane, 0, triangle)
        triangles = new_triangles

    ndc_triangles: list[Triangle] = []
    for triangle in triangles:
        v1: Vertex = perspective_divide(triangle.v1)
        v2: Vertex = perspective_divide(triangle.v2)
        v3: Vertex = perspective_divide(triangle.v3)
        ndc_triangles.append(Triangle(v1, v2, v3))

    return ndc_triangles


def clip_triangle_against_plane(plane_normal: Vec4, plane_distance: float, triangle: Triangle) -> list[Triangle]:
    vertices: list[Vertex] = []
    for i in range(0, 3):
        next_i: int = (i + 1) % 3
        current_distance: float = dot(
            plane_normal, triangle[i].pos) + plane_distance

        next_distance: float = dot(
            plane_normal, triangle[next_i].pos) + plane_distance

        current_inside: bool = current_distance >= 0
        next_inside: bool = next_distance >= 0

        if (current_inside):
            vertices.append(triangle[i])

        if (current_inside != next_inside):
            t: float = -current_distance / (next_distance - current_distance)
            new_vertex: Vertex = interpolate_vertex(
                triangle[i], triangle[next_i], t)
            vertices.append(new_vertex)

    if (len(vertices) == 3):
        return [Triangle(*vertices)]
    if (len(vertices) == 4):
        return [Triangle(*vertices[:3]), Triangle(vertices[2], vertices[3], vertices[0])]

    return []


def interpolate_vertex(va: Vertex, vb: Vertex, t: float) -> Vertex:
    pos: Vec4 = va.pos + ((vb.pos - va.pos) * t)
    if (va.fragment_attributes is None or vb.fragment_attributes is None):
        return Vertex(pos=pos, fragment_attributes=None)

    n_attributes: int = len(va.fragment_attributes)
    attributes: list | type(va.fragment_attributes) = []
    for i in range(0, n_attributes):
        interpolated = va.fragment_attributes[i] + (
            (vb.fragment_attributes[i] - va.fragment_attributes[i]) * t)
        attributes.append(interpolated)

    attributes = type(va.fragment_attributes)(*attributes)

    return Vertex(pos=pos, fragment_attributes=attributes)


def perspective_divide(v: Vertex) -> Vertex:
    """
        w component (camera space depth) is saved.
    """

    fragment_attributes: type(v.fragment_attributes) | None = None
    if (v.fragment_attributes is not None):
        fragment_attributes = \
            type(v.fragment_attributes)(
                *[attrib / v.pos.w for attrib in v.fragment_attributes])

    pos: Vec4 = Vec4(v.pos.x / v.pos.w, v.pos.y /
                     v.pos.w, v.pos.z / v.pos.w, v.pos.w)
    return Vertex(pos=pos, fragment_attributes=fragment_attributes)


def viewport_transform(viewport: Viewport, ndc: Vertex) -> Vertex:
    x_px: float = ((ndc.pos.x + 1.0) * 0.5) * (viewport.width - 1)
    y_px: float = ((ndc.pos.y + 1.0) * 0.5) * (viewport.height - 1)
    new_pos: Vec4 = Vec4(x_px, y_px, ndc.pos.z, ndc.pos.w)
    return Vertex(pos=new_pos, fragment_attributes=ndc.fragment_attributes)
