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


def draw(pipeline: GraphicsPipeline, vertex_buffer: dict[str, list], vertex_shader: VertexShader, fragment_shader: FragmentShader | None, vertex_count: int, vertex_offset: int):
    vertices: list[Vertex] = []
    attribute_fields: list[str] = vertex_shader.Attributes._fields
    for i in range(vertex_offset, vertex_count + vertex_offset):
        attribute_values: list = [vertex_buffer[attribute][i]
                                  for attribute in attribute_fields]

        vertex_attributes = vertex_shader.Attributes(
            *attribute_values)

        shaded_vertex: Vertex = vertex_shader(vertex_attributes)

        ndc_pos = perspective_divide(shaded_vertex.pos)
        screen_space_pos = viewport_transform(
            ndc_pos, pipeline.viewport.width, pipeline.viewport.height)

        vertex: Vertex = Vertex(
            pos=screen_space_pos, fragment_attributes=shaded_vertex.fragment_attributes)
        vertices.append(vertex)

    for i in range(0, len(vertices), 3):
        
        v1: Vertex = vertices[i]
        v2: Vertex = vertices[i + 1]
        v3: Vertex = vertices[i + 2]

        centroid: Vec4 = (v1.pos + v2.pos + v3.pos) / 3
        v1_radius: float = (v1.pos - centroid).magnitude()
        v2_radius: float = (v2.pos - centroid).magnitude()
        v3_radius: float = (v3.pos - centroid).magnitude()
        
        bounding_sphere_radius: float = max(max(v1_radius, v2_radius), v3_radius)
        distance: float = centroid.w

        if (bounding_sphere_radius + distance < 0): 
            continue
        
        clipped_vertices: list[Vertex] = []
        if (abs(bounding_sphere_radius - distance) < 0):
            print("triangle clips against the near plane")
            clipped_vertices = clip_triangle(Vec4(0, 0, 0, 1), 0, v1, v2, v3)
        else: 
            clipped_vertices = [v1, v2, v3]
        
        for j in range(0, len(clipped_vertices), 3):
            c_v1: Vertex = clipped_vertices[j]
            c_v2: Vertex = clipped_vertices[j + 1]
            c_v3: Vertex = clipped_vertices[j + 2]

        rasterize_triangle(
            pipeline.framebuffer, fragment_shader, v1, v2, v3)

def clip_triangle(plane_normal: Vec4, plane_distance: float, v1: Vertex, v2: Vertex, v3: Vertex) -> list[list[Vertex]]: 
    d1: float = 1, dot(plane_normal, v1.pos) + plane_distance
    d2: float = dot(plane_normal, v2.pos) + plane_distance
    d3: float = dot(plane_normal, v3.pos) + plane_distance

    vertices: list[Vec4] = [v1, v2, v3]
    distances: list[float] = [d1, d2, d3]
    in_front: list[int] = []
    for i in range(0, len(distances)):
        if (distances[i] >= 0):
            in_front.append(i)
            
    if (len(in_front) == 3):
        return [v1, v2, v3]
    if (len(in_front) == 0): 
        return []
    if (len(in_front) == 1):
        a: Vertex = vertices[in_front[0]]
        b: Vertex = vertices[(in_front[0] + 1) % 3]
        c: Vertex = vertices[(in_front[0] + 2) % 3]
        
        ab_t: float = (-dot(plane_normal, a) - plane_distance) / dot(plane_normal, (b.pos - a.pos))
        ac_t: float = (-dot(plane_normal, a) - plane_distance) / dot(plane_normal, (c.pos - a.pos))

        ab = interpolate_vertex(a, b, ab_t)
        ac = interpolate_vertex(a, c, ac_t)

        return [a, ab, ac]

    if (len(in_front) == 2):
        b: Vertex = vertices.pop(in_front[1])
        a: Vertex = vertices.pop(in_front[0])
        c: Vertex = vertices[0]

        ac_t: float = (-dot(plane_normal, a) - plane_distance) / dot(plane_normal, (c.pos - a.pos))
        bc_t: float = (-dot(plane_normal, b) - plane_distance) / dot(plane_normal, (c.pos - b.pos))

        ac = interpolate_vertex(a, c, ac_t)
        bc = interpolate_vertex(b, c, bc_t)

        return [a, b, ac, ac, b, bc]

def interpolate_vertex(va: Vertex, vb: Vertex, t: float) -> Vertex:
    n_attributes: int = len(va.fragment_attributes)
    attributes: list | type(va.fragment_attributes) = []
    for i in range(0, n_attributes):
        interpolated = va.fragment_attributes[i] + t * (vb.fragment_attributes[i] - va.fragment_attributes[i])
        attributes.append(interpolated)
        
    attributes = type(va.fragment_attributes)(*attributes)
    pos: Vec4 = va.pos + t * (vb.pos - va.pos)
    
    return Vertex(pos=pos, fragment_attributes=attributes)
        

def perspective_divide(vec: Vec4) -> Vec4:
    """
        w component is saved.
    """
    return Vec4(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w, vec.w)


def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4:
    x_px: float = ((ndc.x + 1.0) * 0.5) * (width - 1)
    y_px: float = ((ndc.y + 1.0) * 0.5) * (height - 1)
    return Vec4(x_px, y_px, ndc.z, ndc.w)
