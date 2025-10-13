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
            rasterize_triangle(
                pipeline.framebuffer, fragment_shader, triangle.v1, triangle.v2, triangle.v3)


def clip_triangle(v1: Vertex, v2: Vertex, v3: Vertex) -> list[Triangle]:
    planes: list[Vec4] = [
            Vec4(0, 0, 1, 0), # near
            Vec4(0, 0, -1, 0), # far
            Vec4(1, 0, 0, 0),  # left
            Vec4(-1, 0, 0, 0), # right
            Vec4(0, -1, 0, 0), # top
            Vec4(0, 1, 0, 0)  # bottom
    ]
    
    distances: list[float] = [
            0, # near
            1, # far
            -1, # left
            1, # right
            1, # top
            -1  # bottom
    ]
    
    # if (v1.pos.z == 0 or v2.pos.z == 0 or v3.pos.z == 0)
    #     centroid: Vec4 = (v1.pos + v2.pos + v3.pos) / 3
    # 
    #     v1_radius: float = (v1.pos - centroid).magnitude()
    #     v2_radius: float = (v2.pos - centroid).magnitude()
    #     v3_radius: float = (v3.pos - centroid).magnitude()
    #     
    #     bounding_sphere_radius: float = max(max(v1_radius, v2_radius), v3_radius)
    #     
    #     near_plane: Vec4 = Vec4(0, 0, 1, 0)
    #     sphere_distance: float = dot(near_plane, centroid)
    #     if (sphere_distance + bounding_sphere_radius < 0):
    #         return []
    #     
    #     clip_triangle(near_plane, 0, v1, v2, v3))
    
    v1 = perspective_divide(v1)
    v2 = perspective_divide(v2)
    v3 = perspective_divide(v3)
    
    triangles: list[Triangle] = [Triangle(v1, v2, v3)]
    
    centroid: Vec4 = (v1.pos + v2.pos + v3.pos) / 3
    
    v1_radius: float = (v1.pos - centroid).magnitude()
    v2_radius: float = (v2.pos - centroid).magnitude()
    v3_radius: float = (v3.pos - centroid).magnitude()
    
    bounding_sphere_radius: float = max(max(v1_radius, v2_radius), v3_radius)
    
    for plane, distance in zip(planes, distances):
        sphere_distance: float = dot(plane, centroid) + distance
        
        if (sphere_distance - bounding_sphere_radius > 0):
            continue
        if (sphere_distance + bounding_sphere_radius < 0):
            return []
        
        new_triangles: list[Triangle] = []
        for triangle in triangles:
            new_triangles += clip_triangle_against_plane(plane, distance, triangle)
        triangles = new_triangles

    return triangles


def clip_triangle_against_plane(plane_normal: Vec4, plane_distance: float, triangle: Triangle) -> list[Triangle]: 
    d1: float = dot(plane_normal, triangle.v1.pos) + plane_distance
    d2: float = dot(plane_normal, triangle.v2.pos) + plane_distance
    d3: float = dot(plane_normal, triangle.v3.pos) + plane_distance

    vertices: list[Vertex] = [triangle.v1, triangle.v2, triangle.v3]
    distances: list[float] = [d1, d2, d3]
    in_front: list[int] = []
    for i in range(0, len(distances)):
        if (distances[i] >= 0):
            in_front.append(i)
            
    if (len(in_front) == 3):
        return [triangle]
    if (len(in_front) == 0): 
        return []
    if (len(in_front) == 1):
        a: Vertex = vertices[in_front[0]]
        b: Vertex = vertices[(in_front[0] + 1) % 3]
        c: Vertex = vertices[(in_front[0] + 2) % 3]
        
        ab_t: float = (-dot(plane_normal, a.pos) - plane_distance) / dot(plane_normal, (b.pos - a.pos))
        ac_t: float = (-dot(plane_normal, a.pos) - plane_distance) / dot(plane_normal, (c.pos - a.pos))

        ab = interpolate_vertex(a, b, ab_t)
        ac = interpolate_vertex(a, c, ac_t)

        return [Triangle(a, ab, ac)]

    if (len(in_front) == 2):
        b: Vertex = vertices.pop(in_front[1])
        a: Vertex = vertices.pop(in_front[0])
        c: Vertex = vertices[0]

        ac_t: float = (-dot(plane_normal, a.pos) - plane_distance) / dot(plane_normal, (c.pos - a.pos))
        bc_t: float = (-dot(plane_normal, b.pos) - plane_distance) / dot(plane_normal, (c.pos - b.pos))

        ac = interpolate_vertex(a, c, ac_t)
        bc = interpolate_vertex(b, c, bc_t)

        return [Triangle(a, b, ac), Triangle(ac, b, bc)]

def interpolate_vertex(va: Vertex, vb: Vertex, t: float) -> Vertex:
    pos: Vec4 = va.pos + ((vb.pos - va.pos) * t)
    if (va.fragment_attributes is None or vb.fragment_attributes is None):
        return Vertex(pos=pos, fragment_attributes=None)

    n_attributes: int = len(va.fragment_attributes)
    attributes: list | type(va.fragment_attributes) = []
    for i in range(0, n_attributes):
        interpolated = va.fragment_attributes[i] + ((vb.fragment_attributes[i] - va.fragment_attributes[i])) * t
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
            type(v.fragment_attributes)(*[attrib / v.pos.w for attrib in v.fragment_attributes])

    pos: Vec4 = Vec4(v.pos.x / v.pos.w, v.pos.y / v.pos.w, v.pos.z / v.pos.w, v.pos.w)
    return Vertex(pos=pos, fragment_attributes=fragment_attributes)


def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4:
    x_px: float = ((ndc.x + 1.0) * 0.5) * (width - 1)
    y_px: float = ((ndc.y + 1.0) * 0.5) * (height - 1)
    return Vec4(x_px, y_px, ndc.z, ndc.w)
